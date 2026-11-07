"""Shared stage execution and sorting/curation steps for v2 orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from spyglass.spikesorting.v2._orchestration.types import StageStatus


def _run_stage(
    stage: str,
    exists: bool,
    work: Callable[[], Any],
    partial: dict[str, Any],
) -> tuple[Any, StageStatus, float]:
    """Time a pipeline stage's ``work()``; classify it; wrap failures.

    ``exists`` (checked before ``work``) makes the status ``"reused"``, else
    ``"computed"``. A failure is re-raised as a chained
    :class:`PipelineStageError` carrying a snapshot of ``partial``.

    Returns ``(work_result, status, seconds)``; ``seconds`` is wall-clock
    time in ``work`` this call.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineStageError

    status: StageStatus = "reused" if exists else "computed"
    start = time.perf_counter()
    try:
        result = work()
    except Exception as exc:  # noqa: BLE001 - re-raised as typed + chained
        # Broad so continue_on_error and partial summaries see every failure.
        raise PipelineStageError(
            stage,
            dict(partial),
            str(exc),
            original_type=type(exc).__name__,
        ) from exc
    return result, status, time.perf_counter() - start


def _populate_once(table, key) -> None:
    """Populate ``key``, tolerating a benign concurrent duplicate.

    The v2 tables opt in to ``PopulateMixin``'s per-key lock, so an
    overlapping run waits and then skips the key. Without the lock, an
    overlapping populate can commit the row first; adopt that winner. A
    duplicate error with the row still absent is re-raised.

    ``key`` must be a full single-row primary key, or the recovery check
    could swallow a genuine duplicate.
    """
    import datajoint as dj

    try:
        table.populate(key, reserve_jobs=False)
    except dj.errors.DuplicateError:
        if not (table & key):
            raise


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

    # Zero units is legitimate on a quiet shank.
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
        # An empty but real curation avoids special-casing a None merge_id.
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
    from spyglass.utils import logger

    # ``curation_id`` is not content-addressed, so "reused" means a root exists.
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
        if curation_description:
            stored = (CurationV2 & curation_key).fetch1("description")
            if stored != curation_description:
                ignored = (
                    "run_v2_pipeline: reused root curation_id="
                    f"{curation_key['curation_id']} of sorting_id="
                    f"{sorting_key['sorting_id']}; it keeps its description "
                    f"{stored!r} and curation_description="
                    f"{curation_description!r} is ignored. Curate a child "
                    "to record a new description."
                )
                logger.warning(ignored)
                warnings_list.append(ignored)
        # A concat curation's synthetic timeline gets no merge row.
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
    # curation_id is reusable after deletion; the UUID pins the generation.
    run_summary["root_curation_uuid"] = (CurationV2 & curation_key).fetch1(
        "curation_uuid"
    )
    return curation_key


def _auto_curation_description(
    pipeline_preset: str, existing: set[str], presets: Mapping
) -> str:
    """Description for the pipeline's auto-labeled child curation.

    Child reuse matches on description, so a twin preset (same metric and
    rule recipes) returns the existing twin's description.

    Parameters
    ----------
    pipeline_preset : str
        The preset of this run.
    existing : set of str
        Descriptions of the root curation's existing children.
    presets : Mapping
        Preset name to preset bundle.

    Returns
    -------
    str
        The description to pass when accepting the evaluation labels.
    """

    def describe(name):
        return f"run_v2_pipeline auto-curation (pipeline_preset={name})"

    own = describe(pipeline_preset)
    if own in existing or pipeline_preset not in presets:
        return own

    def recipes(name):
        bundle = presets[name]
        return (bundle.metric_params_name, bundle.auto_curation_rules_name)

    for name in sorted(presets):
        if (
            name != pipeline_preset
            and recipes(name) == recipes(pipeline_preset)
            and describe(name) in existing
        ):
            return describe(name)
    return own


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
    from spyglass.spikesorting.v2._orchestration.presets import (
        _PIPELINE_PRESETS,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
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
    # "reused" only if the evaluation and the child both pre-existed; the
    # child is known only after the work, so classify afterwards.
    eval_was_populated = bool(CurationEvaluation & eval_key)
    sorting_restriction = {"sorting_id": sorting_key["sorting_id"]}

    def _work():
        _populate_once(CurationEvaluation, eval_key)
        sibling_rows = (CurationV2 & sorting_restriction).fetch(
            "curation_id", "parent_curation_id", "description", as_dict=True
        )
        children_before = {row["curation_id"] for row in sibling_rows}
        description = _auto_curation_description(
            pipeline_preset,
            {
                row["description"]
                for row in sibling_rows
                if row["parent_curation_id"] == curation_key["curation_id"]
            },
            _PIPELINE_PRESETS,
        )
        child = CurationEvaluation().use_evaluation_labels(
            eval_key, description=description, reuse_existing=True
        )
        auto_merge_id = _curation_merge_id(
            child, is_concat=is_concat, warnings_list=warnings_list
        )
        return child, auto_merge_id, children_before

    (
        (child, auto_merge_id, children_before),
        _,
        stage_seconds["auto_curation"],
    ) = _run_stage("auto_curation", False, _work, run_summary)
    child_created = child["curation_id"] not in children_before
    run_summary["auto_curation_status"] = (
        "reused" if eval_was_populated and not child_created else "computed"
    )
    run_summary["curation_evaluation_id"] = eval_key["curation_evaluation_id"]
    # Labels only; every unit is still present.
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

    # One full PK per populate, as _populate_once requires.
    def _populate_member_curations():
        for member_key in member_keys:
            _populate_once(ConcatMemberCuration, member_key)
        member_merge_ids = _concat_member_merge_ids(member_curation_key)
        if len(member_merge_ids) != len(member_keys):
            raise ValueError(
                "run_v2_concat_pipeline: ConcatMemberCuration registration is "
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
        len(ConcatMemberCuration & member_keys) == len(member_keys),
        _populate_member_curations,
        run_summary,
    )
    run_summary["member_merge_ids"] = member_merge_ids
