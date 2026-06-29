"""End-to-end orchestration runners, extracted from ``pipeline.py``.

Behavior-preserving: ``_STAGE_STATUSES``, ``_run_stage``, ``run_v2_pipeline``,
and ``run_v2_pipeline_session`` move here verbatim. ``pipeline.py`` becomes a
thin facade re-exporting these (and every other public name) so user import
paths are unchanged. Top of the run -> preflight -> presets dependency DAG;
also imports the two shared run-summary helpers from ``_pipeline_reporting``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

from spyglass.spikesorting.v2._pipeline_preflight import (
    _resolve_session_sort_group_ids,
    preflight_v2_pipeline,
    preflight_v2_pipeline_session,
)
from spyglass.spikesorting.v2._pipeline_presets import _PIPELINE_PRESETS
from spyglass.spikesorting.v2._pipeline_reporting import (
    _run_metadata,
    _run_warnings,
)
from spyglass.spikesorting.v2._recipe_catalog import DEFAULT_PIPELINE_PRESET
from spyglass.spikesorting.v2._pipeline_types import (
    RunV2PipelineSessionResult,
    RunV2PipelineSummary,
    StageStatus,
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


def run_v2_pipeline(
    nwb_file_name: "str | None" = None,
    sort_group_id: "int | None" = None,
    interval_list_name: "str | None" = None,
    team_name: "str | None" = None,
    pipeline_preset: str = DEFAULT_PIPELINE_PRESET,
    description: str = "",
    require_units: bool = False,
    auto_curate: bool = False,
    preflight: bool = True,
    *,
    concat_session_group_owner: "str | None" = None,
    concat_session_group_name: "str | None" = None,
) -> RunV2PipelineSummary:
    """End-to-end sort in one call: select + populate every stage, then curate.

    Two input modes, exactly one required. Single-session mode (recording ->
    optional artifact detection -> sort -> curation) needs ``nwb_file_name``,
    ``sort_group_id``, ``interval_list_name``, ``team_name``. Concat mode
    (member recordings -> ConcatenatedRecording -> sort -> curation, no artifact
    stage) needs ``concat_session_group_owner`` + ``concat_session_group_name``
    and rejects the single-session fields (member teams come from
    ``SessionGroup.Member``). Supplying both, neither, or part of a mode raises
    ``PipelineInputError``.

    Chains the v2 ``insert_selection`` + ``populate`` calls into one
    call. Idempotent: re-running with the same inputs returns the same
    run summary (same merge_id, same intermediate PKs) without
    duplicating rows.

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
        (using the preset's motion-correction recipe), and sorts the result.
        Requires a concat preset (one whose ``motion_correction_params_name`` is
        set); concat sorts run no artifact detection.
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
    description
        Free-text description passed to ``CurationV2.insert_curation``.
    require_units
        If False (default), a sort that finds zero units still produces
        an EMPTY (but real) curation + merge row, with a loud warning --
        zero units is a legitimate result on a quiet shank, and the empty
        row lets downstream code treat it like any other
        ``SpikeSortingOutput`` row. If True, a zero-unit sort raises
        ``ZeroUnitSortError`` instead (for callers that treat zero units
        as a hard error).
    auto_curate
        If False (default), the run stops at the root curation, so a
        convenience call never silently commits suggested labels. If True,
        the run additionally scores the root curation with the preset's
        ``metric_params_name`` + ``auto_curation_rules_name`` rows
        (``CurationEvaluation``) and materializes a committed child curation
        whose labels ARE the evaluation's verdict. The summary then carries
        ``curation_evaluation_id`` (the suggestion selection PK),
        ``auto_curation_id`` / ``auto_merge_id`` (the materialized child
        curation), and ``auto_curation_status`` (these keys are absent when
        ``auto_curate=False``). ``CurationEvaluation`` builds a whitened PCA
        analyzer, so this adds the heaviest populate of the run.
    preflight
        If True (default), run a fast, read-only prerequisite check before any
        populate; a failure raises ``PreflightError`` (with the exact fix). The
        check is mode-specific:

        - single-session: ``preflight_v2_pipeline`` -- the session / interval /
          team / sort-group rows, the preset's parameter rows, and the sorter
          binary.
        - concat: intentionally minimal -- the ``SessionGroup``, its members,
          and the preset's motion-correction row exist. It does NOT verify the
          preset's preprocessing / sorter rows or the sorter binary, so a
          missing one of those surfaces later as its native error (after the
          member / concat populate), not as a preflight failure.

        Pass ``preflight=False`` to skip the check and attempt the run directly
        (e.g. to see the raw underlying error).

    Returns
    -------
    RunV2PipelineSummary
        Run summary. The sort / curation / merge keys are always present; the
        source-stage keys depend on the input mode.

        Always present:
            ``pipeline_preset``          : the pipeline-preset name
            ``sorting_id``               : SortingSelection PK
            ``curation_id``              : CurationV2 PK
            ``merge_id``                 : SpikeSortingOutput master PK
            ``n_units``                  : unit count (0 on a zero-unit sort)
        Single-session mode adds:
            ``recording_id``             : RecordingSelection PK
            ``artifact_detection_id``    : ArtifactDetectionSelection PK, or
                ``None`` when the preset runs no artifact detection
                (``artifact_detection_params_name`` is ``None``)
        Concat mode adds instead (no artifact stage):
            ``member_recording_ids``     : the per-member RecordingSelection PKs
            ``concat_recording_id``      : ConcatenatedRecording PK
        Downstream consumers should key off ``merge_id``. A zero-unit
        sort yields an empty (but real) curation/merge row, not
        ``None``, so the result is always merge-keyable.

        Plus per-stage observability keys (additive; the keys above are
        unchanged):
            ``*_status`` (one per source/sort/curation stage above -- e.g.
                ``recording_status`` / ``artifact_detection_status`` in
                single-session mode, ``member_recording_status`` /
                ``concat_recording_status`` in concat mode, plus
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
        If ``pipeline_preset`` is not a known name.
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
        installed); the insert helpers translate the would-be
        foreign-key error into this clear message. Run
        ``initialize_v2_defaults()`` first.
    datajoint.errors.IntegrityError
        If an upstream sort group / session / interval list / team does
        not exist when ``preflight=False`` -- the foreign-key violation
        surfaces untranslated. ``preflight=True`` catches these earlier
        as a ``PreflightError`` with the exact fix.
    """
    # Validate the preset DB-free, BEFORE importing the DataJoint table modules
    # (importing them activates @schema and needs a live connection). An unknown
    # or motion-pinned (concat) preset then fails fast with PipelineInputError
    # even when the database is offline, rather than an opaque connection error.
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    # Determine + validate the input mode (database-free, before the table
    # imports). Exactly one COMPLETE mode is required: all single-session fields
    # and no concat fields, or both concat fields and no single-session field
    # (team_name is a single-session field, so it is rejected in concat mode --
    # member teams come from SessionGroup.Member).
    single_fields = (
        nwb_file_name,
        sort_group_id,
        interval_list_name,
        team_name,
    )
    concat_fields = (concat_session_group_owner, concat_session_group_name)
    is_single = all(f is not None for f in single_fields) and not any(
        f is not None for f in concat_fields
    )
    is_concat = all(f is not None for f in concat_fields) and not any(
        f is not None for f in single_fields
    )
    if (
        is_single == is_concat
    ):  # neither complete-and-clean (missing/partial/mixed)
        raise PipelineInputError(
            "run_v2_pipeline requires exactly one input mode: either "
            "single-session fields (nwb_file_name, sort_group_id, "
            "interval_list_name, team_name) or concat fields "
            "(concat_session_group_owner, concat_session_group_name)"
        )

    if pipeline_preset not in _PIPELINE_PRESETS:
        raise PipelineInputError(
            f"run_v2_pipeline: unknown pipeline_preset {pipeline_preset!r}. "
            f"Available pipeline presets: {sorted(_PIPELINE_PRESETS)}. "
            "Call spyglass.spikesorting.v2.pipeline.describe_pipeline_presets() to see "
            "what each preset does, or list_pipeline_presets() for just the names."
        )
    bundle = _PIPELINE_PRESETS[pipeline_preset]

    # The preset's motion field selects the mode it is built for: a motion-pinned
    # preset is a concat preset (motion correction runs on the
    # ConcatenatedRecording path), and a preset without it is single-session.
    if is_single and bundle.motion_correction_params_name is not None:
        raise PipelineInputError(
            f"run_v2_pipeline: pipeline_preset {pipeline_preset!r} pins motion "
            "correction (motion_correction_params_name="
            f"{bundle.motion_correction_params_name!r}), so it targets a "
            "concatenated session group, not the single-session inputs given. "
            "Run it in concat mode (concat_session_group_owner / "
            "concat_session_group_name), or choose a non-concat preset."
        )
    if is_concat and bundle.motion_correction_params_name is None:
        raise PipelineInputError(
            "run_v2_pipeline: concat mode requires a preset that pins motion "
            f"correction, but pipeline_preset {pipeline_preset!r} has none. "
            "Choose a concat preset (one whose motion_correction_params_name is "
            "set; see describe_pipeline_presets())."
        )

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        PreflightError,
        ZeroUnitSortError,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
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
    if preflight and is_single:
        report = preflight_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name=interval_list_name,
            team_name=team_name,
            pipeline_preset=pipeline_preset,
            auto_curate=auto_curate,
        )
        if not report.ok:
            raise PreflightError("\n".join(report.errors))
        # Non-blocking advisories are not errors, but dropping them hides real
        # configuration smells. Log each and thread them into the run summary's
        # ``warnings`` (programmatic access) alongside the per-stage warnings.
        preflight_warnings = list(report.warnings)
        for warning in preflight_warnings:
            logger.warning(f"run_v2_pipeline preflight: {warning}")
    elif preflight and is_concat:
        # Minimal concat preflight: the SessionGroup, its members, and the
        # preset's motion-correction row must exist before the heavy member /
        # concat populate (a fuller concat preflight is future work).
        from spyglass.spikesorting.v2.session_group import (
            MotionCorrectionParameters,
            SessionGroup,
        )

        group_key = {
            "session_group_owner": concat_session_group_owner,
            "session_group_name": concat_session_group_name,
        }
        if not (SessionGroup & group_key):
            raise PreflightError(
                f"run_v2_pipeline: SessionGroup {group_key} does not exist. "
                "Create it with SessionGroup.create_group(...) first."
            )
        if not (SessionGroup.Member & group_key):
            raise PreflightError(
                f"run_v2_pipeline: SessionGroup {group_key} has no members."
            )
        if not (
            MotionCorrectionParameters
            & {
                "motion_correction_params_name": (
                    bundle.motion_correction_params_name
                )
            }
        ):
            raise PreflightError(
                "run_v2_pipeline: MotionCorrectionParameters row "
                f"{bundle.motion_correction_params_name!r} (the concat preset's "
                "motion recipe) is missing. Run initialize_v2_defaults()."
            )

    # Per-stage observability. For each stage: derive computed-vs-reused from
    # an existence check on the output row BEFORE populate, time the
    # populate/insert with a monotonic clock, and on failure raise a stage-
    # aware PipelineStageError carrying the run summary built so far.
    run_summary: dict[str, Any] = {"pipeline_preset": pipeline_preset}
    stage_seconds: dict[str, float] = {}
    warnings_list: list[str] = list(preflight_warnings)

    if is_single:
        # Single-session: recording (+ optional artifact detection) -> sort.
        recording_key = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": int(sort_group_id),
                "interval_list_name": interval_list_name,
                "preprocessing_params_name": bundle.preprocessing_params_name,
                "team_name": team_name,
            }
        )
        (
            _,
            run_summary["recording_status"],
            stage_seconds["recording"],
        ) = _run_stage(
            "recording",
            bool(Recording & recording_key),
            lambda: Recording.populate(recording_key, reserve_jobs=False),
            run_summary,
        )
        run_summary["recording_id"] = recording_key["recording_id"]

        # A None artifact name means the preset runs no artifact detection: skip
        # the ArtifactDetectionSelection/populate stage and sort straight off the
        # recording (no ArtifactDetectionSource row), the form concat also uses.
        if bundle.artifact_detection_params_name is None:
            artifact_detection_id = None
            run_summary["artifact_detection_status"] = "skipped"
            stage_seconds["artifact_detection"] = 0.0
        else:
            artifact_detection_key = ArtifactDetectionSelection.insert_selection(
                {
                    "recording_id": recording_key["recording_id"],
                    "artifact_detection_params_name": bundle.artifact_detection_params_name,
                }
            )
            (
                _,
                run_summary["artifact_detection_status"],
                stage_seconds["artifact_detection"],
            ) = _run_stage(
                "artifact_detection",
                bool(ArtifactDetection & artifact_detection_key),
                lambda: ArtifactDetection.populate(
                    artifact_detection_key, reserve_jobs=False
                ),
                run_summary,
            )
            artifact_detection_id = artifact_detection_key[
                "artifact_detection_id"
            ]
        run_summary["artifact_detection_id"] = artifact_detection_id

        sorting_key = SortingSelection.insert_selection(
            {
                "recording_id": recording_key["recording_id"],
                "sorter": bundle.sorter,
                "sorter_params_name": bundle.sorter_params_name,
                "artifact_detection_id": artifact_detection_id,
            }
        )
    else:
        # Concat: populate each member's Recording, then concatenate, then sort.
        # The concat SortingSelection carries no ArtifactDetectionSource row, so
        # the manifest omits the artifact stage and carries concat_recording_id
        # in place of recording_id.
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecording,
            ConcatenatedRecordingSelection,
            SessionGroup,
        )

        group_key = {
            "session_group_owner": concat_session_group_owner,
            "session_group_name": concat_session_group_name,
        }
        # ConcatenatedRecordingSelection requires every member's Recording to be
        # populated under the preset's preprocessing recipe; build them here so a
        # single concat call is as self-contained as a single-session run. The
        # member-recording build is its own stage so a member populate failure
        # surfaces as a PipelineStageError with timing + partial run summary,
        # the same contract as every other stage.
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
            for member in (SessionGroup.Member & group_key).fetch(as_dict=True)
        ]

        def _populate_member_recordings():
            for key in member_recording_keys:
                if not (Recording & key):
                    Recording.populate(key, reserve_jobs=False)

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

        concat_key = ConcatenatedRecordingSelection.insert_selection(
            {
                "session_group_owner": concat_session_group_owner,
                "session_group_name": concat_session_group_name,
                "preprocessing_params_name": bundle.preprocessing_params_name,
                "motion_correction_params_name": (
                    bundle.motion_correction_params_name
                ),
            }
        )
        (
            _,
            run_summary["concat_recording_status"],
            stage_seconds["concat_recording"],
        ) = _run_stage(
            "concat_recording",
            bool(ConcatenatedRecording & concat_key),
            lambda: ConcatenatedRecording.populate(
                concat_key, reserve_jobs=False
            ),
            run_summary,
        )
        run_summary["concat_recording_id"] = concat_key["concat_recording_id"]

        sorting_key = SortingSelection.insert_selection(
            {
                "concat_recording_id": concat_key["concat_recording_id"],
                "sorter": bundle.sorter,
                "sorter_params_name": bundle.sorter_params_name,
            }
        )
    _, run_summary["sorting_status"], stage_seconds["sorting"] = _run_stage(
        "sorting",
        bool(Sorting & sorting_key),
        lambda: Sorting.populate(sorting_key, reserve_jobs=False),
        run_summary,
    )
    run_summary["sorting_id"] = sorting_key["sorting_id"]

    # Zero units is a legitimate result on a quiet shank. Unless the
    # caller set require_units=True, proceed to build an empty (but real)
    # curation + merge row so the result is merge-keyable like any other.
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

    # Record the now-known stable ``n_units`` and the ``warnings`` before the
    # curation stage runs, so a curation-stage failure's partial run summary
    # carries them (not just the pre-sorting keys).
    run_summary["n_units"] = n_units
    run_summary["warnings"] = warnings_list

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
                description
                or f"run_v2_pipeline pipeline_preset={pipeline_preset}"
            ),
            reuse_existing=True,
        )
        # The CurationV2 part on the merge table is auto-registered inside
        # insert_curation (atomically), so the merge_id read-back is part of
        # the curation stage: a reused root whose registration is missing
        # (e.g. deleted out-of-band) then surfaces as a stage-aware
        # PipelineStageError carrying the partial run summary, not a raw
        # fetch1.
        merge_id = (SpikeSortingOutput.CurationV2 & curation_key).fetch1(
            "merge_id"
        )
        return curation_key, merge_id

    (curation_key, merge_id), curation_status, curation_seconds = _run_stage(
        "curation", curation_exists, _curate_and_register, run_summary
    )
    run_summary["curation_status"] = curation_status
    stage_seconds["curation"] = curation_seconds
    run_summary["curation_id"] = curation_key["curation_id"]
    run_summary["merge_id"] = merge_id

    # Optional auto-curation: only when the caller opts in. Score the root
    # curation with the preset's metric + auto-curation rule rows, then
    # materialize a committed child curation whose labels ARE the evaluation's
    # verdict. The default (auto_curate=False) leaves the run
    # initial-curation-only, so a convenience call never silently commits
    # suggested labels. These keys are added to the summary only when opted in
    # (they are NotRequired), so a default run's summary is unchanged.
    if auto_curate:
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
            CurationEvaluation.populate(eval_key, reserve_jobs=False)
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
            auto_merge_id = (SpikeSortingOutput.CurationV2 & child).fetch1(
                "merge_id"
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
        run_summary["curation_evaluation_id"] = eval_key[
            "curation_evaluation_id"
        ]
        run_summary["auto_curation_id"] = child["curation_id"]
        run_summary["auto_merge_id"] = auto_merge_id

    run_summary["stage_seconds"] = stage_seconds
    return cast(RunV2PipelineSummary, run_summary)


def run_v2_pipeline_session(
    nwb_file_name: str,
    interval_list_name: str,
    team_name: str,
    pipeline_preset: "str | None" = None,
    sort_group_ids: "list[int] | None" = None,
    description: str = "",
    require_units: bool = False,
    auto_curate: bool = False,
    preflight: bool = True,
    continue_on_error: bool = False,
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
    nwb_file_name, interval_list_name, team_name, description, require_units, auto_curate
        As in :func:`run_v2_pipeline`; applied to every group.
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
        A successful entry is the single-group run summary (see
        :func:`run_v2_pipeline`) plus ``sort_group_id`` and ``outcome="ok"``.
        A failed entry is ``{"sort_group_id", "pipeline_preset",
        "outcome": "failed", "error_type", "error", "partial_run_summary",
        "warnings"}`` -- the ``partial_run_summary`` carries the stages
        completed before a sort failure (from :class:`PipelineStageError`) or
        ``None`` when the caught error carries none (a preflight or zero-unit
        failure), and ``warnings`` carries this group's preflight advisories so
        the batch warning count does not under-report failed groups. Wrap with
        ``describe_run(results)`` for a receipt table.

    Raises
    ------
    PipelineInputError
        From the shared target resolver: ``pipeline_preset`` is ``None`` or
        unknown, the session has no sort groups, or a requested
        ``sort_group_ids`` entry is absent. Never suppressed by
        ``continue_on_error``.
    PreflightError
        If ``preflight=True``, a group fails preflight, and
        ``continue_on_error=False`` -- raised before any group is sorted, with
        the aggregated per-group fixes.
    PipelineStageError, ZeroUnitSortError
        Propagated from a per-group :func:`run_v2_pipeline` when
        ``continue_on_error=False``.
    """
    from spyglass.spikesorting.v2.exceptions import (
        PipelineStageError,
        PreflightError,
        ZeroUnitSortError,
    )
    from spyglass.utils import logger

    targets = _resolve_session_sort_group_ids(
        nwb_file_name=nwb_file_name,
        pipeline_preset=pipeline_preset,
        sort_group_ids=sort_group_ids,
        caller="run_v2_pipeline_session",
    )

    results: list[dict[str, Any]] = []
    failed_preflight_ids: set[int] = set()
    preflight_warnings_by_group: dict[int, list[str]] = {}

    # Up-front, read-only whole-session preflight (when requested).
    if preflight:
        session_report = preflight_v2_pipeline_session(
            nwb_file_name=nwb_file_name,
            interval_list_name=interval_list_name,
            team_name=team_name,
            pipeline_preset=pipeline_preset,
            sort_group_ids=targets,
            auto_curate=auto_curate,
        )
        # Capture each group's non-blocking advisories. OK groups run below with
        # preflight=False (the DB checks are not repeated), so without this their
        # preflight warnings would never reach the run summary and the batch
        # warning count would under-report.
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
                    {
                        "sort_group_id": sort_group_id,
                        "pipeline_preset": pipeline_preset,
                        "outcome": "failed",
                        "error_type": "PreflightError",
                        "error": "\n".join(row["errors"]),
                        "partial_run_summary": None,
                        # Carry this group's advisories too, so describe_run /
                        # the batch warning count do not under-report failures.
                        "warnings": list(row.get("warnings", [])),
                    }
                )

    # Per-group compute. Groups covered by the session preflight (or skipped via
    # preflight=False) run with preflight=False so the DB-only checks are not
    # repeated; a per-group failure is caught only for the three pipeline error
    # types, and only when continue_on_error is set.
    for sort_group_id in targets:
        if sort_group_id in failed_preflight_ids:
            continue
        try:
            summary = run_v2_pipeline(
                nwb_file_name=nwb_file_name,
                sort_group_id=sort_group_id,
                interval_list_name=interval_list_name,
                team_name=team_name,
                pipeline_preset=pipeline_preset,
                description=description,
                require_units=require_units,
                auto_curate=auto_curate,
                preflight=False,
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
            results.append(
                {
                    "sort_group_id": sort_group_id,
                    "pipeline_preset": pipeline_preset,
                    "outcome": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "partial_run_summary": getattr(
                        exc, "partial_run_summary", None
                    ),
                    # This group passed preflight (ran with preflight=False) but
                    # failed mid-run; keep its preflight advisories visible. Any
                    # stage warnings live on partial_run_summary, which
                    # _run_warnings reads too.
                    "warnings": preflight_warnings_by_group.get(
                        sort_group_id, []
                    ),
                }
            )
        else:
            # Fold this group's preflight advisories (captured above) into its
            # run summary; the per-group run did no preflight, so there is no
            # overlap with the stage warnings it already carries.
            group_preflight_warnings = preflight_warnings_by_group.get(
                sort_group_id, []
            )
            for warning in group_preflight_warnings:
                logger.warning(
                    "run_v2_pipeline_session: sort_group_id="
                    f"{sort_group_id} preflight: {warning}"
                )
            results.append(
                {
                    **summary,
                    "sort_group_id": sort_group_id,
                    "outcome": "ok",
                    "warnings": list(summary.get("warnings", []))
                    + group_preflight_warnings,
                }
            )

    # Stable, group-ordered result (preflight-failed entries were appended
    # first; restore ascending sort_group_id order).
    results.sort(key=lambda entry: entry["sort_group_id"])

    # End-of-batch receipt in the log. Surfaces the outcomes that are easy to
    # miss when scrolling a long run -- not just failures but zero-unit sorts
    # and warnings too. ``describe_run(results)`` is the richer table form.
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
    return cast(list[RunV2PipelineSessionResult], results)
