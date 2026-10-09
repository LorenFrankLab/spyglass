"""Shared stage execution and sorting/curation steps for v2 orchestration.

Stage timing, failure receipts, and populate concurrency live together here.
The sorting and curation steps compose table operations without duplicating
the scientific computations owned by the tables."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, get_args

from spyglass.spikesorting.v2._core.db_locking import advisory_key_lock
from spyglass.spikesorting.v2._orchestration.types import StageStatus

# Closed vocabulary for stage receipts: work created, reused, or omitted.
_STAGE_STATUSES: frozenset[StageStatus] = frozenset(get_args(StageStatus))


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
    with advisory_key_lock(table, key):
        _populate_tolerating_concurrent_duplicate(table, key)


def _assert_figpack_installed() -> None:
    """Raise ``PipelineInputError`` unless the FigPack packages are installed.

    DB-free, so a run that asks for a FigPack view fails before any table
    import.
    """
    import importlib.util

    from spyglass.spikesorting.v2._review.annotations import (
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
