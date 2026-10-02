"""End-to-end pipeline runners.

Holds ``run_v2_pipeline`` (one sort group or one concat SessionGroup),
``run_v2_pipeline_session`` (every sort group of a session), the motion
``estimate_motion`` entry point, the UnitMatch run/plan helpers, and the
per-stage helpers they share. ``pipeline.py`` re-exports the public names.
Imports ``_pipeline_preflight``, ``_pipeline_presets``, and the run-summary
helpers in ``_pipeline_reporting``; none of those import this module.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, cast

# The best-effort advisory lock lives in the DB-free ``_db_locking`` leaf
# (so domain tables can serialize without importing this orchestration module);
# re-exported under the private names this module's callers + tests already use.
from spyglass.spikesorting.v2._db_locking import (
    POPULATE_LOCK_TIMEOUT_S as _POPULATE_LOCK_TIMEOUT_S,
)
from spyglass.spikesorting.v2._db_locking import (
    advisory_key_lock as _advisory_key_lock,
)

if TYPE_CHECKING:
    import pandas as pd

# DB-free leaf module (imports no schema / no _pipeline_run), so a top-level
# import is cycle-free and keeps the plan annotations resolvable at
# runtime (e.g. typing.get_type_hints for API docs).
from spyglass.spikesorting.v2._unit_match_planning import (
    UnitMatchInputPlan,
    UnitMatchPlan,
)
from spyglass.spikesorting.v2.curation_api import RunResult

from spyglass.spikesorting.v2._pipeline_preflight import (
    _resolve_session_sort_group_ids,
    assert_concat_preflight,
    motion_request_problem,
    preflight_v2_pipeline,
    preflight_v2_pipeline_session,
    resolve_motion_recipe,
    resolve_preset_sort_config,
    supplied_motion_estimate_problem,
)
from spyglass.spikesorting.v2._pipeline_presets import _PIPELINE_PRESETS
from spyglass.spikesorting.v2._pipeline_reporting import (
    _run_metadata,
    _run_warnings,
)
from spyglass.spikesorting.v2._recipe_catalog import DEFAULT_PIPELINE_PRESET
from spyglass.spikesorting.v2._pipeline_types import (
    EstimateMotionReceipt,
    MotionMode,
    RunV2PipelineSessionFailed,
    RunV2PipelineSessionOk,
    RunV2PipelineSessionResult,
    RunV2UnitMatchSummary,
    StageStatus,
    UnitMatchInputSummary,
    UnitMatchMemberChoices,
    UnitMatchStageSeconds,
)

# Closed vocabulary for the per-stage ``*_status`` run-summary keys. A stage is
# ``"computed"`` when its row did not exist before this call and populate /
# insert_curation created it this call; ``"reused"`` when the row already
# existed and the call no-opped; ``"skipped"`` when the preset configured no
# such stage (e.g. a no-artifact preset's artifact_detection stage). Test code
# asserts each status is a member.
_STAGE_STATUSES: frozenset[StageStatus] = frozenset(
    {"computed", "reused", "skipped"}
)


def _run_stage(
    stage: str,
    exists: bool,
    work: Callable[[], Any],
    partial: dict[str, Any],
) -> tuple[Any, StageStatus, float]:
    """Time a pipeline stage's ``work()``; classify it; wrap failures.

    ``exists`` is the result of a pre-``work`` existence check on the stage's
    output row, so the status is ``"reused"`` when the row was already present
    (``work`` no-ops) and ``"computed"`` otherwise. ``work`` is a zero-arg
    closure over the stage's ``populate`` / ``insert_curation`` call; its
    return value is passed back (the curation stage needs the returned key).
    A failure is re-raised as a chained :class:`PipelineStageError` carrying a
    snapshot of ``partial`` (the run summary accumulated from earlier stages)
    so
    the caller sees which stage broke and what was already built.

    Returns ``(work_result, status, seconds)`` where ``seconds`` is monotonic
    wall-clock spent in ``work`` THIS call (≈0 on a reused no-op), not
    cumulative compute cost.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineStageError

    status: StageStatus = "reused" if exists else "computed"
    start = time.perf_counter()
    try:
        result = work()
    except Exception as exc:  # noqa: BLE001 - re-raised as typed + chained
        # Broad on purpose: ANY stage failure must become a typed
        # PipelineStageError so partial_run_summary resume / continue_on_error
        # work. The original type + message are preserved (``original_type``
        # + chained ``from exc``); the catch is NOT narrowed.
        raise PipelineStageError(
            stage,
            dict(partial),
            str(exc),
            original_type=type(exc).__name__,
        ) from exc
    return result, status, time.perf_counter() - start


def _populate_tolerating_concurrent_duplicate(table, key) -> None:
    """Populate ``key``, tolerating a benign concurrent duplicate.

    The pipeline populates with ``reserve_jobs=False`` (no DataJoint ``~jobs``
    reservation), so an overlapping populate of the same *content-addressed* key
    -- another kernel, a lab populate worker, or a re-run of a long stage -- can
    commit the row while this call is still in ``make_compute``, making this
    call's insert raise ``DuplicateError``. Adopt the committed winner (the
    stage is effectively "reused"). A sorter may produce different spikes
    from the same inputs; Sorting keeps each attempt's analyzer private until
    insertion establishes the winner, so the losing compute cannot replace it.

    A duplicate error with the row STILL absent afterward, or any non-duplicate
    error, is a genuine failure and is re-raised unchanged.

    Precondition: ``key`` must be a single-row primary key. The ``table & key``
    recovery check proves that THAT specific row now exists; with an
    under-specified / multi-row restriction a truthy match would no longer prove
    the failed row committed, and a genuine duplicate could be swallowed. Every
    call site passes a full selection PK, so this holds.
    """
    import datajoint as dj

    try:
        table.populate(key, reserve_jobs=False)
    except dj.errors.DuplicateError:
        if not (table & key):
            raise


def _populate_once(table, key) -> None:
    """Populate ``key`` exactly once, even under concurrent/overlapping runs.

    Serializes same-key populate with a self-releasing MySQL advisory lock so
    the expensive compute (e.g. a 70-minute sort) runs once, not once per racing
    run -- WITHOUT ``reserve_jobs=True`` and its stale-``~jobs`` bookkeeping. A
    second run/worker BLOCKS on the lock until the run holding it commits, then
    -- because the row now exists -- DataJoint's own ``key_source - target``
    diff makes its ``populate`` a no-op (no re-compute). ``populate`` is always
    invoked (never short-circuited here), so a genuine stage failure still
    surfaces; and if the lock cannot be taken (timeout / a non-transactional
    server), :func:`_populate_tolerating_concurrent_duplicate` keeps the
    residual insert race from failing the pipeline.

    The "runs once" guarantee is best-effort, not absolute: the lock is bound to
    the DB session, so if that session drops and reconnects mid-compute the lock
    is released and a concurrent run could recompute. The duplicate-tolerance
    still prevents a hard failure -- the worst case degrades to duplicated
    compute. Correctness relies on each stage's publication/insert contract,
    independently of this compute-deduplication lock.
    """
    with _advisory_key_lock(table, key):
        _populate_tolerating_concurrent_duplicate(table, key)


def _validate_run_request(
    caller: str,
    *,
    nwb_file_name,
    sort_group_id,
    interval_list_name,
    team_name,
    concat_session_group_owner,
    concat_session_group_name,
    pipeline_preset: str,
    motion_mode,
    motion_correction_params_name,
    manual_excluded_times,
    motion_estimate_id=None,
) -> tuple[bool, Any, Any, dict]:
    """Validate a run request without touching the database.

    Determines the input mode, checks the preset name and the motion request,
    and folds manual exclusions into the preset's artifact recipe. Runs before
    any DataJoint table import (importing them activates ``@schema`` and needs
    a live connection), so a bad request fails fast even offline.

    Parameters
    ----------
    caller : str
        The public entry point, named in every error message.
    nwb_file_name, sort_group_id, interval_list_name, team_name
        The single-session inputs (all or none).
    concat_session_group_owner, concat_session_group_name
        The concat inputs (both or neither).
    pipeline_preset : str
        A ``_PIPELINE_PRESETS`` name.
    motion_mode, motion_correction_params_name
        The motion request (see :func:`motion_request_problem`).
    manual_excluded_times
        The caller's manual exclusions (intervals, or a member-index mapping
        for concat).
    motion_estimate_id
        A saved estimate to apply (see :func:`motion_request_problem`).

    Returns
    -------
    is_concat : bool
        True for concat mode, False for single-session mode.
    bundle : _PipelinePreset
        The preset, with a ``"none"`` artifact recipe when manual exclusions
        need a detection output and the preset scans for none.
    manual_excluded_times : list or dict
        The normalized manual exclusions.
    source_inputs : dict
        The mode's source fields: ``nwb_file_name``, ``sort_group_id``,
        ``interval_list_name`` and ``team_name``, or
        ``concat_session_group_owner`` and ``concat_session_group_name``.

    Raises
    ------
    PipelineInputError
        On an incomplete or mixed input mode, an unknown preset, or a
        contradictory motion request.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    # Exactly one COMPLETE mode is required: all single-session fields and no
    # concat fields, or both concat fields and no single-session field
    # (team_name is a single-session field, so it is rejected in concat mode --
    # member teams come from SessionGroup.Member).
    single_named = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": interval_list_name,
        "team_name": team_name,
    }
    concat_named = {
        "concat_session_group_owner": concat_session_group_owner,
        "concat_session_group_name": concat_session_group_name,
    }
    single_set = {k for k, v in single_named.items() if v is not None}
    concat_set = {k for k, v in concat_named.items() if v is not None}
    is_single = len(single_set) == len(single_named) and not concat_set
    is_concat = len(concat_set) == len(concat_named) and not single_set

    if not (is_single or is_concat):
        # When the caller clearly started ONE mode but left it incomplete, name
        # the missing field(s) instead of the generic two-mode explanation --
        # the common first-run slip (e.g. forgetting sort_group_id) otherwise
        # misreads as if single-session and concat inputs were mixed.
        if single_set and not concat_set:
            missing = [k for k in single_named if k not in single_set]
            raise PipelineInputError(
                f"{caller}: single-session mode is missing required "
                f"field(s): {', '.join(missing)}. Provide all of "
                f"{', '.join(single_named)}, or switch to concat mode "
                "(concat_session_group_owner + concat_session_group_name)."
            )
        if concat_set and not single_set:
            missing = [k for k in concat_named if k not in concat_set]
            raise PipelineInputError(
                f"{caller}: concat mode is missing required field(s): "
                f"{', '.join(missing)}. Provide both of "
                f"{', '.join(concat_named)}, or switch to single-session mode "
                "(nwb_file_name, sort_group_id, interval_list_name, team_name)."
            )
        # Nothing set, or fields from BOTH modes set (a genuine mode clash):
        # explain the two available input modes.
        raise PipelineInputError(
            f"{caller} requires exactly one input mode: either "
            "single-session fields (nwb_file_name, sort_group_id, "
            "interval_list_name, team_name) or concat fields "
            "(concat_session_group_owner, concat_session_group_name)"
        )

    if pipeline_preset not in _PIPELINE_PRESETS:
        raise PipelineInputError(
            f"{caller}: unknown pipeline_preset {pipeline_preset!r}. "
            f"Available pipeline presets: {sorted(_PIPELINE_PRESETS)}. "
            "Call spyglass.spikesorting.v2.pipeline.describe_pipeline_presets() to see "
            "what each preset does, or list_pipeline_presets() for just the names."
        )
    motion_problem = motion_request_problem(
        motion_mode, motion_correction_params_name, motion_estimate_id
    )
    if motion_problem is not None:
        raise PipelineInputError(f"{caller}: {motion_problem}")
    from spyglass.spikesorting.v2._manual_artifacts import (
        artifact_recipe_with_manual_exclusions,
        resolve_manual_exclusions,
    )

    manual_excluded_times = resolve_manual_exclusions(
        manual_excluded_times, concat=is_concat
    )
    bundle = artifact_recipe_with_manual_exclusions(
        _PIPELINE_PRESETS[pipeline_preset], manual_excluded_times
    )
    source_inputs = concat_named if is_concat else single_named
    return is_concat, bundle, manual_excluded_times, source_inputs


def _run_preflight(
    caller: str,
    *,
    is_concat: bool,
    source_inputs: dict,
    bundle,
    pipeline_preset: str,
    auto_curate: bool,
    manual_excluded_times,
    motion_mode,
    motion_correction_params_name,
    motion_estimate_id=None,
    sort_checks: bool = True,
) -> list[str]:
    """Run the mode's read-only preflight; return its advisories.

    Single-session mode runs the full :func:`preflight_v2_pipeline`; concat
    mode runs :func:`assert_concat_preflight` (the full preflight checks
    single-session rows that do not apply to a concat SessionGroup).

    Parameters
    ----------
    caller : str
        The public entry point, prefixed to each logged advisory and each
        concat preflight error.
    is_concat : bool
        The input mode.
    source_inputs : dict
        The mode's source fields: ``nwb_file_name``, ``sort_group_id``,
        ``interval_list_name`` and ``team_name``, or
        ``concat_session_group_owner`` and ``concat_session_group_name``.
    bundle, pipeline_preset, auto_curate, manual_excluded_times
        As validated by :func:`_validate_run_request`.
    motion_mode, motion_correction_params_name, motion_estimate_id
        The motion request.
    sort_checks : bool
        False skips the checks only a sort needs (see
        :func:`preflight_v2_pipeline`).

    Returns
    -------
    list[str]
        Non-blocking advisories, each also logged.

    Raises
    ------
    PreflightError
        If a prerequisite is missing.
    """
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.utils import logger

    if not is_concat:
        report = preflight_v2_pipeline(
            **source_inputs,
            pipeline_preset=pipeline_preset,
            auto_curate=auto_curate,
            manual_excluded_times=manual_excluded_times,
            motion_mode=motion_mode,
            motion_correction_params_name=motion_correction_params_name,
            motion_estimate_id=motion_estimate_id,
            sort_checks=sort_checks,
        )
        if not report.ok:
            raise PreflightError("\n".join(report.errors))
        # Non-blocking advisories are not errors, but dropping them hides real
        # configuration smells. Log each and thread them into the run summary's
        # ``warnings`` (programmatic access) alongside the per-stage warnings.
        warnings = list(report.warnings)
        for warning in warnings:
            logger.warning(f"{caller} preflight: {warning}")
        return warnings
    # Concat preflight: the SessionGroup + members + each member's
    # raw/valid-times/sort-group/rate prerequisites (+ the preset's
    # auto-curation rows when opted in) + the compute-time param rows and
    # sorter binary, all BEFORE the heavy member / concat populate. Raises
    # PreflightError with the exact fix on the first missing prerequisite.
    return assert_concat_preflight(
        source_inputs["concat_session_group_owner"],
        source_inputs["concat_session_group_name"],
        bundle,
        auto_curate=auto_curate,
        manual_excluded_times=manual_excluded_times,
        motion_mode=motion_mode,
        motion_correction_params_name=motion_correction_params_name,
        motion_estimate_id=motion_estimate_id,
        caller=caller,
        sort_checks=sort_checks,
    )


class _RunSource(NamedTuple):
    """The built source of a run, keyed for the stages that read it.

    Attributes
    ----------
    selection_fields : dict
        The source fields of both ``MotionEstimateSelection`` and
        ``SortingSelection`` (a sort and its motion estimate read one source
        under one mask): ``recording_id`` and ``artifact_detection_id``
        (``None`` without artifact detection), or ``concat_recording_id``.
    concat_key : dict or None
        The ``ConcatenatedRecordingSelection`` PK in concat mode.
    """

    selection_fields: dict
    concat_key: "dict | None"


def _build_run_source(
    *,
    is_concat: bool,
    source_inputs: dict,
    bundle,
    manual_excluded_times,
    run_summary: dict,
    stage_seconds: dict,
) -> _RunSource:
    """Select and populate the source stages a sort reads.

    Single-session: the ``Recording`` and (unless the preset runs none) its
    artifact detection. Concat: each member's ``Recording`` and artifact
    detection, then the ``ConcatenatedRecording``. Records each stage's id,
    status and seconds in ``run_summary`` / ``stage_seconds`` as it goes, so a
    failure's partial summary carries every stage that completed.

    Parameters
    ----------
    is_concat : bool
        The input mode.
    source_inputs : dict
        The mode's source fields (see :func:`_run_preflight`).
    bundle, manual_excluded_times
        As validated by :func:`_validate_run_request`.
    run_summary, stage_seconds : dict
        The run's accumulating summary and per-stage seconds (mutated).

    Returns
    -------
    _RunSource

    Raises
    ------
    PipelineStageError
        If a source stage's populate fails.
    PipelineInputError
        If concat manual exclusions name an absent member.
    """
    if not is_concat:
        return _build_single_session_source(
            source_inputs=source_inputs,
            bundle=bundle,
            manual_excluded_times=manual_excluded_times,
            run_summary=run_summary,
            stage_seconds=stage_seconds,
        )
    return _build_concat_source(
        source_inputs=source_inputs,
        bundle=bundle,
        manual_excluded_times=manual_excluded_times,
        run_summary=run_summary,
        stage_seconds=stage_seconds,
    )


def _build_single_session_source(
    *,
    source_inputs: dict,
    bundle,
    manual_excluded_times,
    run_summary: dict,
    stage_seconds: dict,
) -> _RunSource:
    """Select and populate the ``Recording`` and its artifact detection.

    The artifact-detection stage is ``"skipped"`` when the preset runs none.
    Parameters and return as in :func:`_build_run_source`.
    """
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )

    nwb_file_name = source_inputs["nwb_file_name"]
    sort_group_id = source_inputs["sort_group_id"]
    # Single-session: recording (+ optional artifact detection).
    run_summary["source_mode"] = "single_session"
    recording_key = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(sort_group_id),
            "interval_list_name": source_inputs["interval_list_name"],
            "preprocessing_params_name": bundle.preprocessing_params_name,
            "team_name": source_inputs["team_name"],
        }
    )
    (
        _,
        run_summary["recording_status"],
        stage_seconds["recording"],
    ) = _run_stage(
        "recording",
        bool(Recording & recording_key),
        lambda: _populate_once(Recording, recording_key),
        run_summary,
    )
    run_summary["recording_id"] = recording_key["recording_id"]

    # A None artifact name means the preset runs no artifact detection: skip
    # the RecordingArtifactSelection/populate stage and sort straight off the
    # recording (no ArtifactDetectionSource row), the form concat also uses.
    if bundle.artifact_detection_params_name is None:
        artifact_detection_id = None
        run_summary["artifact_detection_status"] = "skipped"
        stage_seconds["artifact_detection"] = 0.0
    else:
        artifact_detection_key = RecordingArtifactSelection.insert_selection(
            {
                "recording_id": recording_key["recording_id"],
                "artifact_detection_params_name": bundle.artifact_detection_params_name,
                "manual_excluded_times": manual_excluded_times,
            }
        )
        (
            _,
            run_summary["artifact_detection_status"],
            stage_seconds["artifact_detection"],
        ) = _run_stage(
            "artifact_detection",
            bool(RecordingArtifactDetection & artifact_detection_key),
            lambda: _populate_once(
                RecordingArtifactDetection, artifact_detection_key
            ),
            run_summary,
        )
        artifact_detection_id = artifact_detection_key["artifact_detection_id"]
    run_summary["artifact_detection_id"] = artifact_detection_id
    source = {
        "recording_id": recording_key["recording_id"],
        "artifact_detection_id": artifact_detection_id,
    }
    return _RunSource(source, None)


def _build_concat_source(
    *,
    source_inputs: dict,
    bundle,
    manual_excluded_times,
    run_summary: dict,
    stage_seconds: dict,
) -> _RunSource:
    """Build each member's recording and artifact mask, then the concat.

    Parameters, return and errors as in :func:`_build_run_source`.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    # Member detections are inputs to the masked concat.
    run_summary["source_mode"] = "concat"
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
        SessionGroup,
    )

    concat_session_group_owner = source_inputs["concat_session_group_owner"]
    concat_session_group_name = source_inputs["concat_session_group_name"]
    group_key = {
        "session_group_owner": concat_session_group_owner,
        "session_group_name": concat_session_group_name,
    }
    # ConcatenatedRecordingSelection requires every member's Recording to be
    # populated under the preset's preprocessing recipe; build them here so a
    # single concat call is as self-contained as a single-session run. The
    # member-recording build is its own stage so a member populate failure
    # surfaces as a PipelineStageError with timing + partial run summary,
    # the same contract as every other stage. Order by member_index so
    # member_recording_ids is deterministic and matches the concat
    # identity/snapshot ordering, not the implicit DB fetch order.
    members = (SessionGroup.Member & group_key).fetch(
        as_dict=True, order_by="member_index"
    )
    unknown_members = set(manual_excluded_times) - {
        int(m["member_index"]) for m in members
    }
    if unknown_members:
        raise PipelineInputError(
            f"Manual exclusions name absent concat members: {sorted(unknown_members)}"
        )
    member_recording_keys = _build_member_recordings(
        members, bundle, run_summary, stage_seconds
    )
    artifact_ids = _build_member_artifacts(
        members,
        member_recording_keys,
        bundle,
        manual_excluded_times,
        run_summary,
        stage_seconds,
    )

    concat_key = ConcatenatedRecordingSelection.insert_selection(
        {
            "session_group_owner": concat_session_group_owner,
            "session_group_name": concat_session_group_name,
            "preprocessing_params_name": bundle.preprocessing_params_name,
        },
        artifact_detection_ids=artifact_ids,
    )
    (
        _,
        run_summary["concat_recording_status"],
        stage_seconds["concat_recording"],
    ) = _run_stage(
        "concat_recording",
        bool(ConcatenatedRecording & concat_key),
        lambda: _populate_once(ConcatenatedRecording, concat_key),
        run_summary,
    )
    run_summary["concat_recording_id"] = concat_key["concat_recording_id"]
    concat_row = (ConcatenatedRecording & concat_key).fetch1()
    valid_duration = sum(
        end - start for start, end in concat_row["obs_intervals"]
    )
    run_summary["artifact_masked_duration_s"] = float(
        concat_row["total_duration_s"] - valid_duration
    )
    source = {"concat_recording_id": concat_key["concat_recording_id"]}
    return _RunSource(source, dict(concat_key))


def _build_member_recordings(
    members: list[dict],
    bundle,
    run_summary: dict,
    stage_seconds: dict,
) -> list[dict]:
    """Select and populate every concat member's ``Recording`` as one stage.

    Parameters
    ----------
    members : list[dict]
        The ``SessionGroup.Member`` rows, in ``member_index`` order.
    bundle : _PipelinePreset
        The validated preset.
    run_summary, stage_seconds : dict
        The run's accumulating summary and per-stage seconds (mutated).

    Returns
    -------
    list[dict]
        The members' ``RecordingSelection`` keys, in member order.
    """
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )

    member_recording_keys = [
        RecordingSelection.insert_selection(
            {
                "nwb_file_name": member["nwb_file_name"],
                "sort_group_id": int(member["sort_group_id"]),
                "interval_list_name": member["interval_list_name"],
                "preprocessing_params_name": bundle.preprocessing_params_name,
                "team_name": member["team_name"],
            }
        )
        for member in members
    ]

    def _populate_member_recordings():
        for key in member_recording_keys:
            if not (Recording & key):
                _populate_once(Recording, key)

    (
        _,
        run_summary["member_recording_status"],
        stage_seconds["member_recording"],
    ) = _run_stage(
        "member_recording",
        all(bool(Recording & key) for key in member_recording_keys),
        _populate_member_recordings,
        run_summary,
    )
    run_summary["member_recording_ids"] = [
        key["recording_id"] for key in member_recording_keys
    ]
    return member_recording_keys


def _build_member_artifacts(
    members: list[dict],
    member_recording_keys: list[dict],
    bundle,
    manual_excluded_times,
    run_summary: dict,
    stage_seconds: dict,
) -> dict:
    """Select and populate every member's artifact detection as one stage.

    Records each member's detection id, status and masked duration in
    ``run_summary["member_artifacts"]``. The stage is ``"skipped"`` when the
    preset runs no artifact detection.

    Parameters
    ----------
    members : list[dict]
        The ``SessionGroup.Member`` rows, in ``member_index`` order.
    member_recording_keys : list[dict]
        The members' ``RecordingSelection`` keys, in the same order.
    bundle, manual_excluded_times
        As validated by :func:`_validate_run_request` (exclusions keyed by
        ``member_index``).
    run_summary, stage_seconds : dict
        The run's accumulating summary and per-stage seconds (mutated).

    Returns
    -------
    dict
        ``member_index`` to ``artifact_detection_id`` (``None`` without
        artifact detection), the concat selection's member masks.
    """
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import Recording

    artifact_ids = {int(member["member_index"]): None for member in members}
    run_summary["member_artifacts"] = []
    if bundle.artifact_detection_params_name is None:
        run_summary["member_artifact_detection_status"] = "skipped"
        stage_seconds["member_artifact_detection"] = 0.0
    else:
        artifact_keys = [
            RecordingArtifactSelection.insert_selection(
                {
                    **recording_key,
                    "artifact_detection_params_name": bundle.artifact_detection_params_name,
                    "manual_excluded_times": manual_excluded_times.get(
                        int(member["member_index"]), []
                    ),
                }
            )
            for member, recording_key in zip(
                members, member_recording_keys, strict=True
            )
        ]

        def _populate_member_artifacts():
            from spyglass.spikesorting.v2._sorting_artifact_mask import (
                artifact_frame_ranges,
            )

            for member, artifact_key, recording_key in zip(
                members, artifact_keys, member_recording_keys, strict=True
            ):
                reused = bool(RecordingArtifactDetection & artifact_key)
                _populate_once(RecordingArtifactDetection, artifact_key)
                artifact_id = artifact_key["artifact_detection_id"]
                artifact_ids[int(member["member_index"])] = artifact_id
                member_recording = Recording().get_recording(recording_key)
                kept = (
                    RecordingArtifactDetection().get_artifact_removed_intervals(
                        artifact_key
                    )
                )
                # Count frames actually masked, excluding wall-clock gaps.
                excluded = artifact_frame_ranges(
                    member_recording,
                    kept,
                    artifact_detection_id=artifact_id,
                    recording_id=recording_key["recording_id"],
                )
                masked_duration = (
                    sum(end - start for start, end in excluded)
                    / member_recording.get_sampling_frequency()
                )
                run_summary["member_artifacts"].append(
                    {
                        "member_index": int(member["member_index"]),
                        "artifact_detection_id": artifact_id,
                        "status": "reused" if reused else "computed",
                        "masked_duration_s": masked_duration,
                    }
                )

        (
            _,
            run_summary["member_artifact_detection_status"],
            stage_seconds["member_artifact_detection"],
        ) = _run_stage(
            "member_artifact_detection",
            all(
                bool(RecordingArtifactDetection & key) for key in artifact_keys
            ),
            _populate_member_artifacts,
            run_summary,
        )
    return artifact_ids


def _run_motion_estimate(
    source: dict,
    motion_recipe,
    run_summary: dict,
    stage_seconds: dict,
    warnings_list: list,
    motion_estimate_id=None,
) -> dict:
    """Select and populate the motion estimate of a run's source.

    Records the estimate id, its resolved SpikeInterface preset, its status
    and seconds, and its continuity spans without evidence (a non-empty list
    also appends a logged warning) in ``run_summary`` / ``stage_seconds``.

    Parameters
    ----------
    source : dict
        The ``MotionEstimateSelection`` source (``_RunSource.selection_fields``).
    motion_recipe : MotionRecipe
        The resolved ``MotionCorrectionParameters`` recipe.
    run_summary, stage_seconds : dict
        The run's accumulating summary and per-stage seconds (mutated).
    warnings_list : list
        The run's warnings (appended to).
    motion_estimate_id : uuid.UUID or str, optional
        A saved estimate to reuse instead of selecting one: it must be a
        populated estimate of ``source`` made with the recipe's estimation
        row (:func:`supplied_motion_estimate_problem`), and it is never
        recomputed (the stage is ``"reused"``).

    Returns
    -------
    dict
        ``{"motion_estimate_id": ...}``.

    Raises
    ------
    PipelineStageError
        If the selection is refused, the estimation fails, or the supplied
        estimate does not match the source or recipe (stage
        ``"motion_estimate"``).
    """
    from spyglass.spikesorting.v2.motion import (
        MotionEstimate,
        MotionEstimateSelection,
    )
    from spyglass.utils import logger

    def _preset() -> str:
        return (MotionEstimate & estimate_key).fetch1("resolved_params")[
            "preset"
        ]

    if motion_estimate_id is not None:
        from spyglass.spikesorting.v2._source_resolution import SourceLineage

        estimate_key = {
            "motion_estimate_id": uuid.UUID(str(motion_estimate_id))
        }
        if "concat_recording_id" in source:
            lineage = SourceLineage(
                "concatenated_recording", dict(source), None
            )
        else:
            lineage = SourceLineage(
                "recording",
                {"recording_id": source["recording_id"]},
                source["artifact_detection_id"],
            )

        # Checked against the source this run built (for a concat, its member
        # masks too), then read back without any populate.
        def _estimate() -> str:
            problem = supplied_motion_estimate_problem(
                motion_estimate_id, motion_recipe, source_lineage=lineage
            )
            if problem is not None:
                raise ValueError(problem)
            return _preset()

        exists = True
    else:
        # The selection insert runs inside ``_run_stage`` too, so a refused
        # selection (e.g. a source whose content hash drifted) is a
        # PipelineStageError with the partial summary like a failed populate.
        estimate_key, _, _ = _run_stage(
            "motion_estimate",
            False,
            lambda: MotionEstimateSelection.insert_selection(
                {
                    **source,
                    "motion_estimation_params_name": motion_recipe.recipe[
                        "motion_estimation_params_name"
                    ],
                }
            ),
            run_summary,
        )

        # The stage's work populates AND reads back its row, so a row that is
        # missing afterwards is a stage failure with the partial summary.
        def _estimate() -> str:
            _populate_once(MotionEstimate, estimate_key)
            return _preset()

        exists = bool(MotionEstimate & estimate_key)

    (
        run_summary["motion_estimation_preset"],
        run_summary["motion_estimate_status"],
        stage_seconds["motion_estimate"],
    ) = _run_stage("motion_estimate", exists, _estimate, run_summary)
    run_summary["motion_estimate_id"] = estimate_key["motion_estimate_id"]
    # Surfaced, not refused: dropped-frame gaps can leave spans too short to
    # hold a peak, and the estimate there is the temporal prior alone.
    empty_spans = MotionEstimate().get_spans_without_evidence(estimate_key)
    run_summary["motion_spans_without_evidence"] = empty_spans
    if empty_spans:
        empty_span_warning = (
            f"Motion estimate {estimate_key['motion_estimate_id']}: "
            f"{len(empty_spans)} continuity span(s) kept no peaks "
            "(source times "
            + ", ".join(
                f"{span['source_start_s']:.3f}-{span['source_end_s']:.3f} s"
                for span in empty_spans
            )
            + "); the displacement there rests on the estimator's "
            "temporal prior only, so a correction applied to them is not "
            "evidence-based. See run_summary"
            "['motion_spans_without_evidence'] or "
            "MotionEstimate().get_spans_without_evidence(...)."
        )
        logger.warning(empty_span_warning)
        warnings_list.append(empty_span_warning)
    return dict(estimate_key)


def _run_motion_correction(
    estimate_key: dict,
    motion_recipe,
    run_summary: dict,
    stage_seconds: dict,
) -> dict:
    """Select and populate the corrected recording of a saved estimate.

    Records the corrected recording id, the channels its border mode removed,
    its status and seconds in ``run_summary`` / ``stage_seconds``.

    Parameters
    ----------
    estimate_key : dict
        ``{"motion_estimate_id": ...}`` of a populated ``MotionEstimate``.
    motion_recipe : MotionRecipe
        The resolved ``MotionCorrectionParameters`` recipe, whose
        interpolation row is applied.
    run_summary, stage_seconds : dict
        The run's accumulating summary and per-stage seconds (mutated).

    Returns
    -------
    dict
        ``{"motion_corrected_recording_id": ...}``, the key fragment the sort
        selection needs.

    Raises
    ------
    PipelineStageError
        If the selection is refused or the interpolation fails (stage
        ``"motion_corrected_recording"``).
    """
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionCorrectedRecordingSelection,
    )

    corrected_key, _, _ = _run_stage(
        "motion_corrected_recording",
        False,
        lambda: MotionCorrectedRecordingSelection.insert_selection(
            {
                "motion_estimate_id": estimate_key["motion_estimate_id"],
                "motion_interpolation_params_name": motion_recipe.recipe[
                    "motion_interpolation_params_name"
                ],
            }
        ),
        run_summary,
    )

    def _correct() -> list:
        _populate_once(MotionCorrectedRecording, corrected_key)
        return list(
            (MotionCorrectedRecording & corrected_key).fetch1(
                "removed_channel_ids"
            )
        )

    (
        run_summary["motion_removed_channel_ids"],
        run_summary["motion_corrected_recording_status"],
        stage_seconds["motion_corrected_recording"],
    ) = _run_stage(
        "motion_corrected_recording",
        bool(MotionCorrectedRecording & corrected_key),
        _correct,
        run_summary,
    )
    run_summary["motion_corrected_recording_id"] = corrected_key[
        "motion_corrected_recording_id"
    ]
    return dict(corrected_key)


def run_v2_pipeline(
    nwb_file_name: "str | None" = None,
    sort_group_id: "int | None" = None,
    interval_list_name: "str | None" = None,
    team_name: "str | None" = None,
    pipeline_preset: str = DEFAULT_PIPELINE_PRESET,
    curation_description: str = "",
    require_units: bool = False,
    auto_curate: bool = False,
    preflight: bool = True,
    *,
    concat_session_group_owner: "str | None" = None,
    concat_session_group_name: "str | None" = None,
    build_figpack_view: bool = False,
    figpack_label_options: "list[str] | None" = None,
    manual_excluded_times=None,
    motion_mode: MotionMode = "off",
    motion_correction_params_name: "str | None" = None,
    motion_estimate_id: "uuid.UUID | str | None" = None,
) -> "RunResult":
    """End-to-end sort in one call: select + populate every stage, then curate.

    Two input modes, exactly one required. Single-session mode (recording ->
    optional artifact detection -> [motion] -> sort -> curation) needs
    ``nwb_file_name``, ``sort_group_id``, ``interval_list_name``,
    ``team_name``. Concat mode (member recordings -> member artifact masks ->
    ConcatenatedRecording -> [motion] -> sort -> curation) needs
    ``concat_session_group_owner`` + ``concat_session_group_name``
    and rejects the single-session fields (member teams come from
    ``SessionGroup.Member``). Supplying both, neither, or part of a mode raises
    ``PipelineInputError``.

    Chains the v2 ``insert_selection`` + ``populate`` calls into one
    call. Idempotent: re-running with the same inputs reuses the same scientific
    outputs (same root_merge_id and intermediate PKs) without duplicating rows.
    Reported statuses and timings describe the current call.

    Prerequisites (set these up first, in order)
    --------------------------------------------
    1. ``initialize_v2_defaults()`` -- seed the default Lookup rows.
    2. ``LabTeam`` row for ``team_name`` -- the owning team must already
       exist in ``common.LabTeam``.
    3. ``SortGroupV2.set_group_by_shank(nwb_file_name=...)`` (or
       ``set_group_by_electrode_table_column``) -- sort-group structure is
       session-specific user input the orchestrator does not auto-create.

    So a single-session sort is ~4 user touchpoints (the three setup steps
    above plus this call), not 2: this orchestrator collapses the per-stage
    ``insert_selection`` / ``populate`` boilerplate, not the upstream
    session/team/sort-group setup. With ``preflight=True`` (the default)
    this call verifies those prerequisites in ~1 s before any populate and
    raises ``PreflightError`` with the exact fix if one is missing; call
    ``preflight_v2_pipeline(...)`` directly to inspect the report without
    running.

    Parameters
    ----------
    nwb_file_name
        Session whose data will be sorted. The session must already be
        ingested via ``insert_sessions``.
    sort_group_id
        ID of an existing ``SortGroupV2`` row for this session.
        Callers create sort groups via
        ``SortGroupV2.set_group_by_shank`` (or
        ``set_group_by_electrode_table_column``) before calling this
        helper; the orchestrator does not auto-create them because
        sort-group structure is session-specific user input.
    interval_list_name
        Name of the IntervalList row to sort. Typically ``"raw data
        valid times"`` for a full-session sort.
    team_name
        LabTeam owning the sort. Must already exist in
        ``common.LabTeam``. Single-session mode only; rejected in concat mode.
    concat_session_group_owner, concat_session_group_name
        Concat mode (required together, mutually exclusive with the
        single-session fields): the ``(session_group_owner, session_group_name)``
        of an existing ``SessionGroup``. The orchestrator populates each
        member's ``Recording``, concatenates them via ``ConcatenatedRecording``
        (concatenation itself never corrects motion; see ``motion_mode``), and
        sorts the result. Any preset
        runs in either mode; the mode is set by which inputs are given. The
        artifact recipe is applied independently to each member.
    pipeline_preset
        Pipeline-preset name from ``_PIPELINE_PRESETS``. The default is
        ``franklab_probe_hippocampus_30khz_ms5_2026_06`` (MountainSort5),
        which runs under the v2 ``numpy>=2`` baseline out of the box; it is the
        probe-labeled twin of the tetrode-labeled MS5 preset (both resolve to the
        same parameter rows -- ``probe_type`` is informational).

        MountainSort4 is the scientifically-preferred polymer-probe recipe, but
        its ``ml_ms4alg`` backend needs ``numpy<2``, so it is not the default.
        Run it on a modern (``numpy>=2``) host via the containerized
        ``franklab_probe_hippocampus_30khz_ms4_singularity_2026_06`` preset
        (the recommended-science MS4 path when Docker/Singularity is available),
        or on a ``numpy<2`` host via the local
        ``franklab_probe_hippocampus_30khz_ms4_2026_06`` preset; preflight fails
        a selected-but-unrunnable MS4 path with an actionable message.

        Call ``describe_pipeline_presets()`` for a table of what each one does
        (sorter, parameter rows, intended use, and threshold units), or
        ``list_pipeline_presets()`` for just the names.
    curation_description
        Free-text description passed to ``CurationV2.insert_curation``.
    require_units
        If False (default), a sort that finds zero units still produces
        an EMPTY (but real) curation, with a loud warning -- zero units is a
        legitimate result on a quiet shank. A single-session curation also gets
        an empty merge row; a concat curation stays behind the timeline-safety
        gate. If True, a zero-unit sort raises ``ZeroUnitSortError`` instead
        (for callers that treat zero units as a hard error).
    auto_curate
        If False (default), the run stops at the root curation, so a
        convenience call never silently commits suggested labels. If True,
        the run additionally scores the root curation with the preset's
        ``metric_params_name`` + ``auto_curation_rules_name`` rows
        (``CurationEvaluation``) and materializes a committed child curation
        whose labels ARE the evaluation's verdict. The summary then carries
        ``curation_evaluation_id`` (the suggestion selection PK),
        and ``auto_curation_status`` (both absent when ``auto_curate=False``),
        and the always-present ``auto_labeled_curation_id`` /
        ``auto_labeled_curation_uuid`` name the materialized child;
        ``auto_labeled_merge_id`` is also set for a single-session run and
        remains ``None`` for concat. All stay ``None`` on a root-only run. Automatic labels are suggestions written as labels,
        not approval, and the child still holds EVERY unit -- select the
        analysis population explicitly with ``select_units_for_analysis``.
        ``CurationEvaluation`` builds a whitened PCA analyzer, so this adds
        the heaviest populate of the run.
    preflight
        If True (default), run a fast, read-only prerequisite check before any
        populate; a failure raises ``PreflightError`` (with the exact fix). The
        check is mode-specific:

        - single-session: ``preflight_v2_pipeline`` -- the session / interval /
          team / sort-group rows, the preset's parameter rows, and the sorter
          binary.
        - concat: ``assert_concat_preflight`` -- the ``SessionGroup`` and its
          members, plus the preset's preprocessing / sorter / analyzer-waveform
          param rows and the sorter binary/runtime (the compute-row checks
          shared with the single-session preflight), including the member
          artifact parameters.

        Pass ``preflight=False`` to skip the check and attempt the run directly
        (e.g. to see the raw underlying error).
    build_figpack_view
        If True, additionally publish an offline FigPack manual-curation view of
        the run's ROOT curation and add its local bundle URI to the summary
        (``figpack_uri``) along with a ``figpack`` stage. Default ``False``.
        Requires the optional FigPack packages (the ``spikesorting-v2-curation``
        extra); ``build_figpack_view=True`` without them fails fast with
        ``PipelineInputError`` before any populate. A zero-unit sort has no
        analyzer to summarize, so the view is skipped (``figpack_status`` is
        ``"skipped"`` and ``figpack_uri`` is absent) rather than failing the run.
        Hosted upload is not offered here -- the bundle is always local.
    manual_excluded_times
        Immutable half-open [start, stop) exclusions in original session
        seconds, composed with automatic artifact detection. For concat,
        map each member_index to its intervals. Manual exclusions also apply
        when the preset disables automatic detection.
    figpack_label_options
        Curation label palette (in display order) for the FigPack view; passed
        through to ``FigPackCurationSelection``. ``None`` (default) uses
        ``["accept", "mua", "noise"]``. Ignored when
        ``build_figpack_view=False``.
    motion_mode
        The motion stage, in either input mode, run on the sort's source (the
        recording under its artifact mask, or the concatenation):

        - ``"off"`` (default): no motion stage; the sort reads the masked,
          uncorrected source.
        - ``"estimate"``: also save a ``MotionEstimate`` of that source (for
          QC). The sort is exactly the ``"off"`` sort -- same ``sorting_id``,
          same traces.
        - ``"apply"``: save the estimate and a ``MotionCorrectedRecording``,
          and sort the corrected recording (a different ``sorting_id``).

        Experimental: no motion recipe is validated for a probe. An
        estimation or application failure raises ``PipelineStageError`` for
        that stage and no sort is attempted; the run never falls back to the
        uncorrected source. With ``"apply"``, a ``SorterParameters`` row that
        runs the sorter's own motion correction is rejected (preflight names
        the key to turn off).
    motion_correction_params_name
        The ``MotionCorrectionParameters`` recipe (an estimation recipe plus
        an interpolation recipe; ``initialize_v2_defaults`` ships
        ``dredge_v1`` and ``dredge_fast_v1``). Required iff ``motion_mode``
        is not ``"off"``. A recipe with ``"off"``, no recipe with
        ``"estimate"`` / ``"apply"``, or an unknown mode raises
        ``PipelineInputError`` before any database access. A recipe name with
        no ``MotionCorrectionParameters`` row fails preflight
        (``PreflightError``); with ``preflight=False`` it raises
        ``ValueError`` (like any missing parameter row) before any populate.
    motion_estimate_id
        With ``motion_mode="apply"`` only: apply exactly this saved
        ``MotionEstimate`` instead of selecting the source's estimate. It is
        reused as is (never recomputed) and corrected with the interpolation
        row of ``motion_correction_params_name``. It must be a populated
        estimate of this run's source and artifact mask (for concat: its
        session group, preprocessing recipe, members and member masks) made
        with that recipe's estimation row, on source traces unchanged since;
        a mismatch names the estimate's value and the run's, and fails
        preflight (``PreflightError``) or, with ``preflight=False``, the
        ``motion_estimate`` stage (``PipelineStageError``), before any
        sort. Given with another
        mode, or not a UUID, it raises ``PipelineInputError`` before any
        database access.

    Returns
    -------
    RunResult
        Mapping-compatible run summary wrapping a
        ``RunV2SingleSessionSummary`` or ``RunV2ConcatSummary``. In addition to
        the preserved item keys it exposes ``root_curation`` and
        ``auto_labeled_curation`` generation-pinned accessors (built from the
        ``*_curation_uuid`` keys recorded at run time). A concat run keeps its
        synthetic-timeline root/analysis merge IDs unset and instead returns one
        session-safe merge ID per frozen member. The source-stage keys depend on
        the input mode, discriminated by ``source_mode``.

        Always present:
            ``pipeline_preset``          : the pipeline-preset name
            ``source_mode``              : ``"single_session"`` or ``"concat"``
                (the discriminant for the mode-specific source keys below)
            ``sorting_id``               : SortingSelection PK
            ``root_curation_id``         : the ROOT (uncurated) CurationV2 PK
            ``root_merge_id``            : the root's SpikeSortingOutput PK;
                ``None`` for a concat run
            ``auto_labeled_curation_id``     : the auto-labeled child CurationV2
                PK, or ``None`` on a root-only run
            ``auto_labeled_merge_id``        : that child's SpikeSortingOutput
                PK, or ``None`` on a root-only or concat run
            ``root_curation_uuid`` / ``auto_labeled_curation_uuid`` : the
                generation UUIDs the ``root_curation`` / ``auto_labeled_curation``
                accessors are pinned to
            ``sorter_config``            : what the sort stage executes
                (``EffectiveSortConfig.as_dict()``: SI kwargs, whiten routing,
                seed, job kwargs, backend)
            ``n_units``                  : unit count (0 on a zero-unit sort)
        Single-session mode adds:
            ``recording_id``             : RecordingSelection PK
            ``artifact_detection_id``    : RecordingArtifactSelection PK, or
                ``None`` when the preset runs no artifact detection
                (``artifact_detection_params_name`` is ``None``)
        Concat mode adds member artifact detection and concat stages:
            ``member_recording_ids``     : the per-member RecordingSelection PKs
            ``concat_recording_id``      : ConcatenatedRecording PK
            ``member_merge_ids``         : frozen ``member_index`` to
                wall-clock-aligned SpikeSortingOutput PK; points to the
                auto-curated child when ``auto_curate=True``, otherwise the root
        Motion keys (always present; ``None`` where they do not apply):
            ``motion_mode`` / ``motion_correction_params_name`` : the request
            ``motion_estimate_id``       : MotionEstimateSelection PK
                (``"estimate"`` / ``"apply"``)
            ``motion_estimate_supplied`` : whether the caller supplied that
                estimate (``motion_estimate_id=``) rather than the run
                selecting the source's estimate
            ``motion_estimation_preset`` : the SpikeInterface preset the
                estimation recipe resolved to
            ``motion_corrected_recording_id`` :
                MotionCorrectedRecordingSelection PK (``"apply"``)
            ``motion_removed_channel_ids`` : source channels the interpolation's
                ``remove_channels`` border mode dropped (``"apply"``; empty
                for ``force_extrapolate``)
            ``motion_spans_without_evidence`` : the estimate's continuity
                spans that kept no peak
                (``MotionEstimate.get_spans_without_evidence``; ``"estimate"``
                / ``"apply"``; empty when every span has evidence). A
                non-empty list also adds a ``warnings`` entry.
        ``build_figpack_view=True`` adds (unless the sort found zero units):
            ``figpack_uri``              : the published FigPack curation-view
                URI (a local bundle path; offline only)
        Neither merge id is a filtered unit set: the auto-labeled child still
        carries every unit, labels included. Hand a curation to analysis with
        ``select_units_for_analysis(run.auto_labeled_curation, policy=...)``
        (or the root / a manually curated child), which builds the
        ``SortedSpikesGroup`` downstream reads and reports the included /
        excluded unit ids. For concat sorts that helper uses the per-member
        session-timeline rows. There is deliberately no bare ``merge_id``. A zero-unit single-session sort yields an empty (but real)
        root curation/merge row. A concat run leaves only its unsafe synthetic-
        timeline merge IDs ``None``; its member IDs are session-safe.

        Plus per-stage observability keys (additive; the keys above are
        unchanged):
            ``*_status`` (one per source/sort/curation stage above -- e.g.
                ``recording_status`` / ``artifact_detection_status`` in
                single-session mode, ``member_recording_status`` /
                ``concat_recording_status`` / ``member_curation_status`` in
                concat mode, plus
                ``sorting_status`` / ``curation_status``) : ``"computed"`` if the
                stage did work this call, ``"reused"`` if its row already existed
                and the call no-opped, or ``"skipped"`` if the preset configured
                no such stage (only ``artifact_detection_status`` for a
                no-artifact preset) -- see ``_STAGE_STATUSES``.
            ``stage_seconds``     : dict of monotonic wall-clock seconds spent
                per stage **this call** (the same stage names as the ``*_status``
                keys above) -- ≈0 on an idempotent re-run, NOT cumulative
                compute cost.
            ``warnings``          : list of human-readable advisories raised
                during the run (e.g. the zero-unit message); empty when
                clean.
        Two identical calls return equal run summaries except for
        ``stage_seconds`` and the ``*_status`` values (the second reports
        ``"reused"``), inserting no duplicate rows.

    Raises
    ------
    PipelineInputError
        If ``pipeline_preset`` is not a known name, or the motion request is
        contradictory (see ``motion_mode`` and ``motion_estimate_id``).
    PreflightError
        If ``preflight=True`` and a prerequisite is missing (the message
        lists every failed check and its fix). Bypass with
        ``preflight=False``.
    PipelineStageError
        If a compute stage's ``populate`` / ``insert_curation`` fails. Names
        the failing stage and carries the partial run summary of the stages
        that completed before it (the original error is chained). Only the
        compute
        stages are wrapped; an error from the cheap ``insert_selection``
        prelude surfaces as its own native exception (e.g.
        ``DuplicateSelectionError``).
    ZeroUnitSortError
        If the sort finds zero units and ``require_units=True``.
    ValueError
        If a required parameter Lookup row is missing (e.g.
        ``PreprocessingParameters`` / ``SorterParameters`` defaults not
        installed, or the ``MotionCorrectionParameters`` recipe when
        ``preflight=False``); the insert helpers translate the would-be
        foreign-key error into this clear message. Run
        ``initialize_v2_defaults()`` first.
    datajoint.errors.IntegrityError
        If an upstream sort group / session / interval list / team does
        not exist when ``preflight=False`` -- the foreign-key violation
        surfaces untranslated. ``preflight=True`` catches these earlier
        as a ``PreflightError`` with the exact fix.
    """
    # Validate the request DB-free, BEFORE importing the DataJoint table modules
    # (importing them activates @schema and needs a live connection). An unknown
    # preset, an incomplete input mode or a contradictory motion request then
    # fails fast with PipelineInputError even when the database is offline,
    # rather than an opaque connection error.
    is_concat, bundle, manual_excluded_times, source_inputs = (
        _validate_run_request(
            "run_v2_pipeline",
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name=interval_list_name,
            team_name=team_name,
            concat_session_group_owner=concat_session_group_owner,
            concat_session_group_name=concat_session_group_name,
            pipeline_preset=pipeline_preset,
            motion_mode=motion_mode,
            motion_correction_params_name=motion_correction_params_name,
            manual_excluded_times=manual_excluded_times,
            motion_estimate_id=motion_estimate_id,
        )
    )

    # Fail fast (still DB-free, before the table imports) if a FigPack view was
    # requested without the optional packages installed -- otherwise the missing
    # install would surface only as an opaque import error after a full sort.
    if build_figpack_view:
        _assert_figpack_installed()

    # Import the merge, sorting, and curation table modules after the DB-free
    # checks and before preflight, so a schema that cannot activate fails the
    # run before any compute. The names are unused here (each stage helper
    # imports what it uses); the auto-curation (``metric_curation``) and
    # FigPack (``_figpack_curation``) modules are imported only by their
    # stages.
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import (
        CONCAT_MERGE_GATE_MESSAGE,
        CurationV2,
    )
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.exceptions import ZeroUnitSortError
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )
    from spyglass.utils import logger

    # Fail fast: a read-only config check before any insert/populate. Single-
    # session mode runs the full preflight; concat mode runs a minimal one (the
    # full preflight checks single-session rows that do not apply to a concat
    # SessionGroup). Bypass either with preflight=False.
    preflight_warnings: list[str] = []
    if preflight:
        preflight_warnings = _run_preflight(
            "run_v2_pipeline",
            is_concat=is_concat,
            source_inputs=source_inputs,
            bundle=bundle,
            pipeline_preset=pipeline_preset,
            auto_curate=auto_curate,
            manual_excluded_times=manual_excluded_times,
            motion_mode=motion_mode,
            motion_correction_params_name=motion_correction_params_name,
            motion_estimate_id=motion_estimate_id,
        )

    # Per-stage observability. For each stage: derive computed-vs-reused from
    # an existence check on the output row BEFORE populate, time the
    # populate/insert with a monotonic clock, and on failure raise a stage-
    # aware PipelineStageError carrying the run summary built so far.
    run_summary: dict[str, Any] = {
        "pipeline_preset": pipeline_preset,
        "motion_mode": motion_mode,
        "motion_correction_params_name": motion_correction_params_name,
        "motion_estimate_id": None,
        "motion_estimate_supplied": motion_estimate_id is not None,
        "motion_corrected_recording_id": None,
        "motion_estimation_preset": None,
        "motion_removed_channel_ids": None,
        "motion_spans_without_evidence": None,
    }
    # Resolve the motion recipe before any populate, so a missing row fails
    # here (not after the recording / concat build) when preflight is off.
    motion_recipe = (
        None
        if motion_mode == "off"
        else resolve_motion_recipe(motion_correction_params_name)
    )
    # Capture what the sort stage executes ONCE, up front, from the same
    # resolver the dispatcher uses (``resolve_sort_config``): the receipt then
    # states the effective sorter kwargs / whiten routing / seed / job kwargs /
    # backend regardless of whether the stage is computed or reused this call.
    # ``None`` only when the preset's SorterParameters row is absent (the
    # preflight above already failed, or preflight=False bypassed it).
    run_summary["sorter_config"] = resolve_preset_sort_config(bundle)
    stage_seconds: dict[str, float] = {}
    # Point the run summary at the live stage_seconds dict NOW (not only at the
    # end) so a PipelineStageError's partial run summary -- a shallow copy --
    # carries the timing of every stage that completed before the failure, not
    # an empty dict.
    run_summary["stage_seconds"] = stage_seconds
    warnings_list: list[str] = list(preflight_warnings)

    # The scientific setup states the rows the run's stages execute for every
    # sort group its sort reads (a concat's members, in member order).
    from spyglass.spikesorting.v2._pipeline_preflight import (
        describe_scientific_setup,
    )

    run_summary["scientific_config"] = describe_scientific_setup(
        bundle,
        _run_sort_group_keys(
            is_concat,
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            concat_session_group_owner=concat_session_group_owner,
            concat_session_group_name=concat_session_group_name,
        ),
        run_summary["sorter_config"],
        manual_excluded_times=manual_excluded_times,
        concat=is_concat,
        motion_mode=motion_mode,
        motion_recipe=motion_recipe,
    )
    source = _build_run_source(
        is_concat=is_concat,
        source_inputs=source_inputs,
        bundle=bundle,
        manual_excluded_times=manual_excluded_times,
        run_summary=run_summary,
        stage_seconds=stage_seconds,
    )
    # The motion stages run on the sort's source. Only ``"apply"`` changes
    # what the sort reads (the corrected recording), so an ``"estimate"``
    # sort is exactly the ``"off"`` sort. A motion stage failure raises
    # PipelineStageError before any sort.
    corrected: dict = {}
    if motion_recipe is not None:
        estimate_key = _run_motion_estimate(
            source.selection_fields,
            motion_recipe,
            run_summary,
            stage_seconds,
            warnings_list,
            motion_estimate_id=motion_estimate_id,
        )
        if motion_mode == "apply":
            corrected = _run_motion_correction(
                estimate_key, motion_recipe, run_summary, stage_seconds
            )
    sorting_key, n_units = _run_sorting_stage(
        source.selection_fields,
        corrected,
        bundle,
        require_units=require_units,
        run_summary=run_summary,
        stage_seconds=stage_seconds,
        warnings_list=warnings_list,
    )

    # Record the now-known stable ``n_units`` and the ``warnings`` before the
    # curation stage runs, so a curation-stage failure's partial run summary
    # carries them (not just the pre-sorting keys).
    run_summary["n_units"] = n_units
    run_summary["warnings"] = warnings_list

    curation_key = _run_root_curation_stage(
        sorting_key,
        curation_description=curation_description,
        pipeline_preset=pipeline_preset,
        is_concat=is_concat,
        run_summary=run_summary,
        stage_seconds=stage_seconds,
        warnings_list=warnings_list,
    )
    # Auto-labeled pointer: None until something curates the root. A default
    # (root-only) run leaves these None on purpose, so downstream code can't
    # silently decode the uncurated root. ``auto_curate=True`` fills them below.
    run_summary["auto_labeled_curation_id"] = None
    run_summary["auto_labeled_merge_id"] = None
    run_summary["auto_labeled_curation_uuid"] = None

    # Optional auto-curation: only when the caller opts in. Score the root
    # curation with the preset's metric + auto-curation rule rows, then
    # materialize a committed child curation whose labels ARE the evaluation's
    # verdict and point the canonical auto-labeled keys (initialized to None
    # above) at it. The evaluation id and stage status are added only when
    # opted in (NotRequired keys).
    if auto_curate:
        _run_auto_curation_stage(
            sorting_key,
            curation_key,
            bundle,
            pipeline_preset=pipeline_preset,
            is_concat=is_concat,
            run_summary=run_summary,
            stage_seconds=stage_seconds,
            warnings_list=warnings_list,
        )

    # A concat curation's own synthetic-timeline row remains gated, but its
    # final curation for this run (the auto-curated child when present,
    # otherwise the root) is materialized into one session-safe merge row per
    # frozen member.
    if is_concat:
        _run_member_curation_stage(
            sorting_key, source.concat_key, run_summary, stage_seconds
        )

    # Optional FigPack manual-curation view: only when the caller opts in.
    # Publish an OFFLINE FigPack bundle of the ROOT curation (FigPack publishes
    # raw-namespace curations only -- an auto-curated child lives in the
    # curation_evaluation namespace) and surface its local URI. A zero-unit sort
    # has no analyzer to summarize, so the FigPack view is skipped with a warning rather
    # than failing an otherwise-successful empty sort. These keys are added only
    # when opted in (NotRequired), so a default run's summary is unchanged.
    if build_figpack_view:
        _run_figpack_stage(
            sorting_key,
            curation_key,
            n_units=n_units,
            figpack_label_options=figpack_label_options,
            run_summary=run_summary,
            stage_seconds=stage_seconds,
            warnings_list=warnings_list,
        )

    run_summary["stage_seconds"] = stage_seconds
    return RunResult(run_summary)


def _assert_figpack_installed() -> None:
    """Raise ``PipelineInputError`` unless the FigPack packages are installed.

    DB-free, so a run that asks for a FigPack view fails before any table
    import.
    """
    import importlib.util

    from spyglass.spikesorting.v2._figpack_curation import (
        FIGPACK_INSTALL_HINT,
    )
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    if (
        importlib.util.find_spec("figpack") is None
        or importlib.util.find_spec("figpack_spike_sorting") is None
    ):
        raise PipelineInputError(
            "run_v2_pipeline: build_figpack_view=True but the FigPack "
            f"packages are not installed. {FIGPACK_INSTALL_HINT}"
        )


def _run_sort_group_keys(
    is_concat: bool,
    *,
    nwb_file_name,
    sort_group_id,
    concat_session_group_owner,
    concat_session_group_name,
) -> list[dict]:
    """The ``(nwb_file_name, sort_group_id)`` keys a run's sort reads.

    One key for a single-session run; a concat's members, in member order.
    """
    if is_concat:
        from spyglass.spikesorting.v2.session_group import SessionGroup

        group_keys = (
            SessionGroup.Member
            & {
                "session_group_owner": concat_session_group_owner,
                "session_group_name": concat_session_group_name,
            }
        ).fetch(
            "nwb_file_name",
            "sort_group_id",
            as_dict=True,
            order_by="member_index",
        )
    else:
        group_keys = [
            {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}
        ]
    return group_keys


def _curation_merge_id(curation_key, *, is_concat: bool, warnings_list: list):
    """Return one merge id, or the intentional concat ``None``.

    A concat curation has no merge row by design; its gate advisory is
    appended to ``warnings_list`` once.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CONCAT_MERGE_GATE_MESSAGE

    merge_ids = (SpikeSortingOutput.CurationV2 & curation_key).fetch("merge_id")
    if len(merge_ids) == 1:
        return merge_ids[0]
    if len(merge_ids) > 1:
        raise ValueError(
            "run_v2_pipeline: CurationV2 has multiple "
            f"SpikeSortingOutput rows for {curation_key}."
        )
    if not is_concat:
        raise ValueError(
            "run_v2_pipeline: single-session CurationV2 is missing its "
            f"SpikeSortingOutput registration for {curation_key}."
        )
    warning = CONCAT_MERGE_GATE_MESSAGE.format(
        sorting_id=curation_key["sorting_id"]
    )
    if warning not in warnings_list:
        warnings_list.append(warning)
    return None


def _concat_member_merge_ids(curation_key) -> dict[int, Any]:
    """Return the complete frozen-member-index to merge-id mapping."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )

    rows = (
        SpikeSortingOutput.ConcatMemberCuration * ConcatMemberCuration
        & curation_key
    ).fetch("merge_id", "member_index", as_dict=True)
    return {int(row["member_index"]): row["merge_id"] for row in rows}


def _run_sorting_stage(
    source_fields: dict,
    corrected: dict,
    bundle,
    *,
    require_units: bool,
    run_summary: dict,
    stage_seconds: dict,
    warnings_list: list,
) -> tuple[dict, int]:
    """Select and populate the sort; return its key and unit count.

    Parameters
    ----------
    source_fields : dict
        The sort's source (``_RunSource.selection_fields``).
    corrected : dict
        ``{"motion_corrected_recording_id": ...}`` for an ``"apply"`` run,
        else empty.
    bundle : _PipelinePreset
        The validated preset.
    require_units : bool
        Raise on a zero-unit sort instead of warning.
    run_summary, stage_seconds : dict
        The run's accumulating summary and per-stage seconds (mutated).
    warnings_list : list
        The run's warnings (appended to on a zero-unit sort).

    Returns
    -------
    sorting_key : dict
        The ``SortingSelection`` key.
    n_units : int
        The sort's unit count.

    Raises
    ------
    PipelineStageError
        If the sort's populate fails.
    ZeroUnitSortError
        If the sort finds zero units and ``require_units`` is True.
    """
    from spyglass.spikesorting.v2.exceptions import ZeroUnitSortError
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )
    from spyglass.utils import logger

    sorting_key = SortingSelection.insert_selection(
        {
            **source_fields,
            "sorter": bundle.sorter,
            "sorter_params_name": bundle.sorter_params_name,
            **corrected,
        }
    )
    _, run_summary["sorting_status"], stage_seconds["sorting"] = _run_stage(
        "sorting",
        bool(Sorting & sorting_key),
        lambda: _populate_once(Sorting, sorting_key),
        run_summary,
    )
    run_summary["sorting_id"] = sorting_key["sorting_id"]

    # Zero units is a legitimate result on a quiet shank. Unless the caller set
    # require_units=True, proceed to build an empty (but real) curation. A
    # single-session curation remains merge-keyable; a concat curation is gated.
    n_units = int((Sorting & sorting_key).fetch1("n_units"))
    if n_units == 0:
        sorting_id = sorting_key["sorting_id"]
        if require_units:
            raise ZeroUnitSortError(
                "run_v2_pipeline: sort found zero units for "
                f"sorting_id={sorting_id}; require_units=True. Check "
                "detect_threshold / the artifact mask, or call with "
                "require_units=False to accept the empty result."
            )
        # Fall through to the normal curation + merge insert. A
        # zero-unit sort yields an EMPTY (but real) curation + merge row
        # so downstream consumers treat it like any other
        # SpikeSortingOutput row instead of special-casing a None
        # merge_id. The warning is both logged (console) and recorded on
        # the run summary's ``warnings`` list (programmatic access).
        zero_unit_warning = (
            f"run_v2_pipeline: zero units for sorting_id={sorting_id}; "
            "writing an EMPTY curation + merge row. Check "
            "detect_threshold / the artifact mask if you expected output."
        )
        logger.warning(zero_unit_warning)
        warnings_list.append(zero_unit_warning)
    return sorting_key, n_units


def _run_root_curation_stage(
    sorting_key: dict,
    *,
    curation_description: str,
    pipeline_preset: str,
    is_concat: bool,
    run_summary: dict,
    stage_seconds: dict,
    warnings_list: list,
) -> dict:
    """Insert (or reuse) the sort's root curation and register its merge row.

    Records the root curation id, merge id (``None`` for concat) and
    generation UUID in ``run_summary``.

    Returns
    -------
    dict
        The root ``CurationV2`` key.

    Raises
    ------
    PipelineStageError
        If the curation insert or its merge registration fails.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    # Idempotent curation: ``insert_curation`` owns the root-reuse logic.
    # With ``reuse_existing=True`` it returns the canonical (lowest
    # curation_id) existing root if one is present -- deterministically,
    # and through the same source-part / guard / merge-registration path a
    # fresh insert uses -- otherwise it stages a fresh root. Routing through
    # it (rather than a raw fetch-or-insert here) avoids bypassing that
    # guard and silently reusing a root whose description/labels differ.
    # ``curation_id`` is not content-addressed, so classify reused/computed
    # from whether a root curation already exists for this sorting (the same
    # check insert_curation's root-reuse path uses).
    curation_exists = bool(
        CurationV2
        & {"sorting_id": sorting_key["sorting_id"], "parent_curation_id": -1}
    )

    def _curate_and_register():
        curation_key = CurationV2.insert_curation(
            sorting_key=sorting_key,
            labels={},
            parent_curation_id=-1,
            description=(
                curation_description
                or f"run_v2_pipeline pipeline_preset={pipeline_preset}"
            ),
            reuse_existing=True,
        )
        # A single-session curation is registered atomically. A concat curation
        # deliberately is not: its synthetic timeline is unsafe for a
        # session-scoped merge consumer, so its merge id is None and the gate
        # advisory is recorded in the run summary. Any missing single-session
        # registration remains a stage-aware failure.
        merge_id = _curation_merge_id(
            curation_key, is_concat=is_concat, warnings_list=warnings_list
        )
        return curation_key, merge_id

    (curation_key, merge_id), curation_status, curation_seconds = _run_stage(
        "curation", curation_exists, _curate_and_register, run_summary
    )
    run_summary["curation_status"] = curation_status
    stage_seconds["curation"] = curation_seconds
    run_summary["root_curation_id"] = curation_key["curation_id"]
    run_summary["root_merge_id"] = merge_id
    # Pin the generation: the numeric curation_id is reusable after deletion,
    # so the receipt records the row UUID the ``root_curation`` accessor checks.
    run_summary["root_curation_uuid"] = (CurationV2 & curation_key).fetch1(
        "curation_uuid"
    )
    return curation_key


def _run_auto_curation_stage(
    sorting_key: dict,
    curation_key: dict,
    bundle,
    *,
    pipeline_preset: str,
    is_concat: bool,
    run_summary: dict,
    stage_seconds: dict,
    warnings_list: list,
) -> None:
    """Evaluate the root curation and accept its labels into a child.

    Records ``auto_curation_status``, ``curation_evaluation_id`` and the
    ``auto_labeled_*`` keys of the accepted child in ``run_summary``.

    Raises
    ------
    PipelineStageError
        If the evaluation, the label acceptance or the child's merge
        registration fails (stage ``"auto_curation"``).
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import PipelineStageError
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )

    eval_key = CurationEvaluationSelection.insert_selection(
        {
            "sorting_id": sorting_key["sorting_id"],
            "curation_id": curation_key["curation_id"],
            "metric_params_name": bundle.metric_params_name,
            "auto_curation_rules_name": bundle.auto_curation_rules_name,
        }
    )
    # The stage does two things: populate the evaluation AND accept its
    # labels into a committed child curation. Classify it honestly --
    # "reused" only when BOTH were already done (the evaluation was
    # populated and the accepted child already existed), else "computed" --
    # so a run that (re)created the child is never mislabeled "reused" just
    # because the evaluation pre-existed. ``_run_stage``'s exists-before-work
    # model cannot express that (the child id is only known after
    # acceptance), so time + wrap the stage here, preserving the same
    # stage-aware ``PipelineStageError``.
    eval_was_populated = bool(CurationEvaluation & eval_key)
    sorting_restriction = {"sorting_id": sorting_key["sorting_id"]}
    auto_start = time.perf_counter()
    try:
        _populate_once(CurationEvaluation, eval_key)
        children_before = set(
            (CurationV2 & sorting_restriction).fetch("curation_id")
        )
        child = CurationEvaluation().use_evaluation_labels(
            eval_key,
            description=(
                "run_v2_pipeline auto-curation "
                f"(pipeline_preset={pipeline_preset})"
            ),
            reuse_existing=True,
        )
        auto_merge_id = _curation_merge_id(
            child, is_concat=is_concat, warnings_list=warnings_list
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as typed + chained
        raise PipelineStageError(
            "auto_curation",
            dict(run_summary),
            str(exc),
            original_type=type(exc).__name__,
        ) from exc
    stage_seconds["auto_curation"] = time.perf_counter() - auto_start
    child_created = child["curation_id"] not in children_before
    run_summary["auto_curation_status"] = (
        "reused" if eval_was_populated and not child_created else "computed"
    )
    run_summary["curation_evaluation_id"] = eval_key["curation_evaluation_id"]
    # The auto-curated child is the run's auto-labeled curation: labels
    # only, every unit still present, no unit selection applied.
    run_summary["auto_labeled_curation_id"] = child["curation_id"]
    run_summary["auto_labeled_merge_id"] = auto_merge_id
    run_summary["auto_labeled_curation_uuid"] = (CurationV2 & child).fetch1(
        "curation_uuid"
    )


def _run_member_curation_stage(
    sorting_key: dict,
    concat_key: dict,
    run_summary: dict,
    stage_seconds: dict,
) -> None:
    """Register a concat run's final curation once per frozen member.

    The final curation is the auto-curated child when present, otherwise the
    root. Records ``member_curation_status`` and ``member_merge_ids`` in
    ``run_summary``.

    Raises
    ------
    PipelineStageError
        If a member populate fails or a member merge row is missing (stage
        ``"member_curation"``).
    """
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    member_curation_key = {
        "sorting_id": sorting_key["sorting_id"],
        "curation_id": (
            run_summary["auto_labeled_curation_id"]
            if run_summary["auto_labeled_curation_id"] is not None
            else run_summary["root_curation_id"]
        ),
    }
    member_indices = [
        int(index)
        for index in (
            ConcatenatedRecordingSelection.MemberSnapshot & concat_key
        ).fetch("member_index", order_by="member_index")
    ]
    member_keys = [
        {**member_curation_key, "member_index": member_index}
        for member_index in member_indices
    ]

    # Populate each full member PK separately so the advisory lock and
    # benign-duplicate recovery retain their single-key guarantee.
    def _populate_member_curations():
        for member_key in member_keys:
            _populate_once(ConcatMemberCuration, member_key)
        member_merge_ids = _concat_member_merge_ids(member_curation_key)
        if len(member_merge_ids) != len(member_keys):
            raise ValueError(
                "run_v2_pipeline: ConcatMemberCuration registration is "
                f"incomplete for {member_curation_key}: expected "
                f"{len(member_keys)} member merge rows, found "
                f"{len(member_merge_ids)}."
            )
        return member_merge_ids

    (
        member_merge_ids,
        run_summary["member_curation_status"],
        stage_seconds["member_curation"],
    ) = _run_stage(
        "member_curation",
        all(bool(ConcatMemberCuration & key) for key in member_keys),
        _populate_member_curations,
        run_summary,
    )
    run_summary["member_merge_ids"] = member_merge_ids


def _run_figpack_stage(
    sorting_key: dict,
    curation_key: dict,
    *,
    n_units: int,
    figpack_label_options,
    run_summary: dict,
    stage_seconds: dict,
    warnings_list: list,
) -> None:
    """Publish an offline FigPack view of the root curation.

    A zero-unit sort skips the stage with a logged warning. Otherwise
    records ``figpack_status`` and ``figpack_uri`` in ``run_summary``.

    Raises
    ------
    PipelineStageError
        If the FigPack populate fails (stage ``"figpack"``).
    """
    from spyglass.utils import logger

    if n_units == 0:
        figpack_skip_warning = (
            "run_v2_pipeline: build_figpack_view=True but the sort found "
            f"zero units (sorting_id={sorting_key['sorting_id']}); "
            "skipping the FigPack view -- there is no analyzer to summarize."
        )
        logger.warning(figpack_skip_warning)
        warnings_list.append(figpack_skip_warning)
        run_summary["figpack_status"] = "skipped"
        stage_seconds["figpack"] = 0.0
    else:
        from pathlib import Path

        from spyglass.spikesorting.v2.figpack_curation import (
            FigPackCuration,
            FigPackCurationSelection,
        )

        figpack_selection = FigPackCurationSelection.insert_selection(
            {
                "sorting_id": sorting_key["sorting_id"],
                "curation_id": curation_key["curation_id"],
            },
            label_options=figpack_label_options,
            upload=False,
        )
        # Reuse only when the FigPackCuration row AND its offline bundle are
        # both present. The run returns figpack_uri as an output, so a row
        # whose local bundle was cleaned out (e.g. the figpack / temp dir was
        # purged) must be rebuilt rather than reported "reused" with a dead
        # path: drop the stale row so populate rebuilds the bundle.
        built = FigPackCuration & figpack_selection
        bundle_present = (
            bool(built) and Path(built.fetch1("figpack_uri")).exists()
        )
        if bool(built) and not bundle_present:
            built.delete(safemode=False)
        (
            _,
            run_summary["figpack_status"],
            stage_seconds["figpack"],
        ) = _run_stage(
            "figpack",
            bundle_present,
            # Same concurrency handling as every other stage: serialize the
            # populate on the content-addressed selection and tolerate a
            # benign duplicate. (The stale-bundle delete above runs outside
            # the lock -- it only fires when the bundle is already gone, so
            # it cannot race a live build.)
            lambda: _populate_once(FigPackCuration, figpack_selection),
            run_summary,
        )
        run_summary["figpack_uri"] = (
            FigPackCuration & figpack_selection
        ).fetch1("figpack_uri")


def estimate_motion(
    nwb_file_name: "str | None" = None,
    sort_group_id: "int | None" = None,
    interval_list_name: "str | None" = None,
    team_name: "str | None" = None,
    *,
    pipeline_preset: str = DEFAULT_PIPELINE_PRESET,
    preflight: bool = True,
    concat_session_group_owner: "str | None" = None,
    concat_session_group_name: "str | None" = None,
    manual_excluded_times=None,
    motion_correction_params_name: "str | None" = None,
) -> "EstimateMotionReceipt":
    """Save the motion estimate of a run's source, without sorting.

    Builds the source exactly as ``run_v2_pipeline`` does with the same
    arguments -- the recording and its artifact detection, or the member
    recordings, member artifact masks and their concatenation -- and saves
    its ``MotionEstimate`` with the estimation row of
    ``motion_correction_params_name``. Nothing is sorted or curated. Inspect
    the estimate (``MotionEstimate().get_motion`` /
    ``get_displacement_on_source_clock`` / ``get_spans_without_evidence``),
    then sort the recording corrected with exactly that estimate::

        receipt = estimate_motion(..., motion_correction_params_name=name)
        run_v2_pipeline(
            ...,
            motion_mode="apply",
            motion_correction_params_name=name,
            motion_estimate_id=receipt["motion_estimate_id"],
        )

    Idempotent: the estimate id is content-addressed (source, its content,
    mask, estimation row and resolved configuration), so a second call, or a
    ``run_v2_pipeline`` run in ``"estimate"`` / ``"apply"`` mode on the same
    source and recipe, reuses it.

    Experimental: no motion recipe is validated for a probe. Inspect the
    saved estimate rather than trusting a corrected sort's improvement.

    Parameters
    ----------
    nwb_file_name, sort_group_id, interval_list_name, team_name
        Single-session mode, as in :func:`run_v2_pipeline`. The only
        positional parameters; every other one is keyword-only.
    concat_session_group_owner, concat_session_group_name
        Concat mode, as in :func:`run_v2_pipeline` (mutually exclusive with
        the single-session fields).
    pipeline_preset
        The preset whose preprocessing and artifact rows build the source; use
        the preset of the run that will apply the estimate.
    preflight
        If True (default), run ``run_v2_pipeline``'s read-only preflight for
        ``motion_mode="estimate"`` first, without its sorter-only checks: the
        source prerequisites, the preset's preprocessing and artifact rows,
        the recipe, a filtering preprocessing recipe and a geometry the
        estimation recipe supports. The preset's sorter need not be available
        here (the run that applies the estimate checks it). A failure raises
        ``PreflightError``.
    manual_excluded_times
        Manual exclusions, as in :func:`run_v2_pipeline`; they are part of
        the mask the estimate is made under.
    motion_correction_params_name
        The ``MotionCorrectionParameters`` recipe (required). Its estimation
        row is estimated here; its interpolation row is the one the later
        ``"apply"`` run uses.

    Returns
    -------
    EstimateMotionReceipt
        ``motion_estimate_id``, the resolved SpikeInterface estimation preset,
        the continuity spans without evidence, ``motion_diagnostics`` (peaks
        detected and kept, largest absolute displacement in um, number of
        temporal bins), the source-stage ids and statuses, ``stage_seconds``
        and ``warnings`` (a span without evidence adds one).

    Raises
    ------
    PipelineInputError
        If ``motion_correction_params_name`` is missing, the input mode is
        incomplete or mixed, or ``pipeline_preset`` is unknown -- before any
        database access.
    PreflightError
        If ``preflight=True`` and a prerequisite is missing.
    PipelineStageError
        If a source stage or the estimation fails; names the stage and
        carries the partial summary.
    ValueError
        If the recipe row is missing and ``preflight=False``.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    if motion_correction_params_name is None:
        raise PipelineInputError(
            "estimate_motion requires motion_correction_params_name, a "
            "MotionCorrectionParameters row (e.g. 'dredge_fast_v1'; see "
            "MotionCorrectionParameters()). Its estimation row is estimated "
            "here and its interpolation row is applied by the later "
            "run_v2_pipeline(motion_mode='apply', motion_estimate_id=...)."
        )
    is_concat, bundle, manual_excluded_times, source_inputs = (
        _validate_run_request(
            "estimate_motion",
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name=interval_list_name,
            team_name=team_name,
            concat_session_group_owner=concat_session_group_owner,
            concat_session_group_name=concat_session_group_name,
            pipeline_preset=pipeline_preset,
            motion_mode="estimate",
            motion_correction_params_name=motion_correction_params_name,
            manual_excluded_times=manual_excluded_times,
        )
    )
    warnings_list: list[str] = []
    if preflight:
        warnings_list = _run_preflight(
            "estimate_motion",
            is_concat=is_concat,
            source_inputs=source_inputs,
            bundle=bundle,
            pipeline_preset=pipeline_preset,
            auto_curate=False,
            manual_excluded_times=manual_excluded_times,
            motion_mode="estimate",
            motion_correction_params_name=motion_correction_params_name,
            sort_checks=False,
        )
    from spyglass.spikesorting.v2.motion import MotionEstimate

    stage_seconds: dict[str, float] = {}
    run_summary: dict[str, Any] = {
        "pipeline_preset": pipeline_preset,
        "motion_correction_params_name": motion_correction_params_name,
        "motion_estimate_id": None,
        "motion_estimation_preset": None,
        "motion_spans_without_evidence": None,
        "stage_seconds": stage_seconds,
        "warnings": warnings_list,
    }
    motion_recipe = resolve_motion_recipe(motion_correction_params_name)
    source = _build_run_source(
        is_concat=is_concat,
        source_inputs=source_inputs,
        bundle=bundle,
        manual_excluded_times=manual_excluded_times,
        run_summary=run_summary,
        stage_seconds=stage_seconds,
    )
    estimate_key = _run_motion_estimate(
        source.selection_fields,
        motion_recipe,
        run_summary,
        stage_seconds,
        warnings_list,
    )
    diagnostics = (
        (MotionEstimate & estimate_key)
        .proj(
            "n_peaks_detected",
            "n_peaks_kept",
            "max_abs_displacement_um",
            "n_temporal_bins",
        )
        .fetch1()
    )
    run_summary["motion_diagnostics"] = {
        "n_peaks_detected": int(diagnostics["n_peaks_detected"]),
        "n_peaks_kept": int(diagnostics["n_peaks_kept"]),
        "max_abs_displacement_um": float(
            diagnostics["max_abs_displacement_um"]
        ),
        "n_temporal_bins": int(diagnostics["n_temporal_bins"]),
    }
    return cast("EstimateMotionReceipt", run_summary)


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
            # stage warnings live on partial_run_summary, which
            # _run_warnings reads too.
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

    n_ok = sum(entry["outcome"] == "ok" for entry in results)
    n_failed = sum(entry["outcome"] == "failed" for entry in results)
    n_zero = 0
    n_warn = 0
    failed_details = []
    for entry in results:
        partial = (
            entry.get("partial_run_summary")
            if isinstance(entry.get("partial_run_summary"), dict)
            else {}
        )
        n_units = _run_metadata(entry, partial, "n_units")
        if n_units == 0:
            n_zero += 1
        if _run_warnings(entry, partial):
            n_warn += 1
        if entry["outcome"] == "failed":
            error_type = entry.get("error_type") or "Error"
            failed_details.append(
                f"sort_group_id={entry['sort_group_id']}: {error_type}"
            )
    failed_suffix = f" ({', '.join(failed_details)})" if failed_details else ""
    logger.info(
        f"run_v2_pipeline_session: {len(results)} group(s): {n_ok} ok, "
        f"{n_failed} failed{failed_suffix}, {n_zero} zero-unit, "
        f"{n_warn} with warnings. "
        "Call describe_run(results) for the per-group receipt."
    )


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
    from spyglass.spikesorting.v2._unit_match_planning import (
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
    from spyglass.spikesorting.v2._unit_match_planning import (
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
    from spyglass.spikesorting.v2._unit_match_planning import (
        member_choices_to_dataframe,
    )

    return member_choices_to_dataframe(
        _unit_match_member_choices(session_group_owner, session_group_name)
    )
