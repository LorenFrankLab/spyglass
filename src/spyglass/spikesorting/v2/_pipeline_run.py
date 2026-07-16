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
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pandas as pd

# DB-free leaf module (imports no schema / no _pipeline_run), so a top-level
# import is cycle-free and keeps the UnitMatchPlan annotations resolvable at
# runtime (e.g. typing.get_type_hints for API docs).
from spyglass.spikesorting.v2._unit_match_planning import UnitMatchPlan

from spyglass.spikesorting.v2._pipeline_preflight import (
    _resolve_session_sort_group_ids,
    assert_concat_preflight,
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
    RunV2UnitMatchSummary,
    StageStatus,
    UnitMatchMemberChoices,
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
    curation_description: str = "",
    require_units: bool = False,
    auto_curate: bool = False,
    preflight: bool = True,
    *,
    concat_session_group_owner: "str | None" = None,
    concat_session_group_name: "str | None" = None,
    build_figpack_view: bool = False,
    figpack_label_options: "list[str] | None" = None,
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
    run summary (same root_merge_id, same intermediate PKs) without
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
    curation_description
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
        ``auto_curate=False``); the always-present ``analysis_curation_id`` /
        ``analysis_merge_id`` are set to that child (they stay ``None`` on a
        root-only run). ``CurationEvaluation`` builds a whitened PCA analyzer,
        so this adds the heaviest populate of the run.
    preflight
        If True (default), run a fast, read-only prerequisite check before any
        populate; a failure raises ``PreflightError`` (with the exact fix). The
        check is mode-specific:

        - single-session: ``preflight_v2_pipeline`` -- the session / interval /
          team / sort-group rows, the preset's parameter rows, and the sorter
          binary.
        - concat: ``assert_concat_preflight`` -- the ``SessionGroup``, its
          members, and the preset's motion-correction row, plus the preset's
          preprocessing / sorter / analyzer-waveform param rows and the sorter
          binary/runtime (the compute-row checks shared with the single-session
          preflight; a concat sort runs no artifact stage).

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
    figpack_label_options
        Curation label palette (in display order) for the FigPack view; passed
        through to ``FigPackCurationSelection``. ``None`` (default) uses
        ``["accept", "mua", "noise"]``. Ignored when
        ``build_figpack_view=False``.

    Returns
    -------
    RunV2PipelineSummary
        Run summary -- a ``RunV2SingleSessionSummary`` or ``RunV2ConcatSummary``
        (the two arms of the ``RunV2PipelineSummary`` union). The sort / curation
        / merge keys are always present; the source-stage keys depend on the
        input mode, discriminated by ``source_mode``.

        Always present:
            ``pipeline_preset``          : the pipeline-preset name
            ``source_mode``              : ``"single_session"`` or ``"concat"``
                (the discriminant for the mode-specific source keys below)
            ``sorting_id``               : SortingSelection PK
            ``root_curation_id``         : the ROOT (uncurated) CurationV2 PK
            ``root_merge_id``            : the root's SpikeSortingOutput PK
            ``analysis_curation_id``     : the analysis-ready CurationV2 PK, or
                ``None`` on a root-only run (curate first); equals
                ``auto_curation_id`` when ``auto_curate=True``
            ``analysis_merge_id``        : the analysis-ready SpikeSortingOutput
                PK, or ``None`` on a root-only run; equals ``auto_merge_id``
                when ``auto_curate=True``
            ``n_units``                  : unit count (0 on a zero-unit sort)
        Single-session mode adds:
            ``recording_id``             : RecordingSelection PK
            ``artifact_detection_id``    : ArtifactDetectionSelection PK, or
                ``None`` when the preset runs no artifact detection
                (``artifact_detection_params_name`` is ``None``)
        Concat mode adds instead (no artifact stage):
            ``member_recording_ids``     : the per-member RecordingSelection PKs
            ``concat_recording_id``      : ConcatenatedRecording PK
        ``build_figpack_view=True`` adds (unless the sort found zero units):
            ``figpack_uri``              : the published FigPack curation-view
                URI (a local bundle path; offline only)
        For downstream science key off ``analysis_merge_id`` (the curated,
        analysis-ready handle) -- NOT ``root_merge_id``, the uncurated root.
        There is deliberately no bare ``merge_id``: a default run has no
        analysis-ready id to copy. A zero-unit sort yields an empty (but real)
        root curation/merge row, not ``None``, so the root is always
        merge-keyable.

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
                "run_v2_pipeline: single-session mode is missing required "
                f"field(s): {', '.join(missing)}. Provide all of "
                f"{', '.join(single_named)}, or switch to concat mode "
                "(concat_session_group_owner + concat_session_group_name)."
            )
        if concat_set and not single_set:
            missing = [k for k in concat_named if k not in concat_set]
            raise PipelineInputError(
                "run_v2_pipeline: concat mode is missing required field(s): "
                f"{', '.join(missing)}. Provide both of "
                f"{', '.join(concat_named)}, or switch to single-session mode "
                "(nwb_file_name, sort_group_id, interval_list_name, team_name)."
            )
        # Nothing set, or fields from BOTH modes set (a genuine mode clash):
        # explain the two available input modes.
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

    # Fail fast (still DB-free, before the table imports) if a FigPack view was
    # requested without the optional packages installed -- otherwise the missing
    # install would surface only as an opaque import error after a full sort.
    if build_figpack_view:
        import importlib.util

        from spyglass.spikesorting.v2._figpack_curation import (
            FIGPACK_INSTALL_HINT,
        )

        if (
            importlib.util.find_spec("figpack") is None
            or importlib.util.find_spec("figpack_spike_sorting") is None
        ):
            raise PipelineInputError(
                "run_v2_pipeline: build_figpack_view=True but the FigPack "
                f"packages are not installed. {FIGPACK_INSTALL_HINT}"
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
        # Concat preflight: the SessionGroup + members + each member's
        # raw/valid-times/sort-group/rate prerequisites + the preset's
        # motion-correction row (+ auto-curation rows when opted in) + the
        # compute-time param rows and sorter binary (no artifact stage), all
        # BEFORE the heavy member / concat populate. Raises PreflightError with
        # the exact fix on the first missing prerequisite.
        preflight_warnings = assert_concat_preflight(
            concat_session_group_owner,
            concat_session_group_name,
            bundle,
            auto_curate=auto_curate,
        )

    # Per-stage observability. For each stage: derive computed-vs-reused from
    # an existence check on the output row BEFORE populate, time the
    # populate/insert with a monotonic clock, and on failure raise a stage-
    # aware PipelineStageError carrying the run summary built so far.
    run_summary: dict[str, Any] = {"pipeline_preset": pipeline_preset}
    stage_seconds: dict[str, float] = {}
    # Point the run summary at the live stage_seconds dict NOW (not only at the
    # end) so a PipelineStageError's partial run summary -- a shallow copy --
    # carries the timing of every stage that completed before the failure, not
    # an empty dict.
    run_summary["stage_seconds"] = stage_seconds
    warnings_list: list[str] = list(preflight_warnings)

    if is_single:
        # Single-session: recording (+ optional artifact detection) -> sort.
        run_summary["source_mode"] = "single_session"
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
        run_summary["source_mode"] = "concat"
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
        # the same contract as every other stage. Order by member_index so
        # member_recording_ids is deterministic and matches the concat
        # identity/snapshot ordering, not the implicit DB fetch order.
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
            for member in (SessionGroup.Member & group_key).fetch(
                as_dict=True, order_by="member_index"
            )
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
                curation_description
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
    run_summary["root_curation_id"] = curation_key["curation_id"]
    run_summary["root_merge_id"] = merge_id
    # Analysis-ready pointer: None until something curates the root. A default
    # (root-only) run leaves these None on purpose, so downstream code can't
    # silently decode the uncurated root. ``auto_curate=True`` fills them below.
    run_summary["analysis_curation_id"] = None
    run_summary["analysis_merge_id"] = None

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
        # The auto-curated child IS the analysis-ready curation for this run.
        run_summary["analysis_curation_id"] = child["curation_id"]
        run_summary["analysis_merge_id"] = auto_merge_id

    # Optional FigPack manual-curation view: only when the caller opts in.
    # Publish an OFFLINE FigPack bundle of the ROOT curation (FigPack publishes
    # raw-namespace curations only -- an auto-curated child lives in the
    # curation_evaluation namespace) and surface its local URI. A zero-unit sort
    # has no analyzer to summarize, so the FigPack view is skipped with a warning rather
    # than failing an otherwise-successful empty sort. These keys are added only
    # when opted in (NotRequired), so a default run's summary is unchanged.
    if build_figpack_view:
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
                lambda: FigPackCuration.populate(figpack_selection),
                run_summary,
            )
            run_summary["figpack_uri"] = (
                FigPackCuration & figpack_selection
            ).fetch1("figpack_uri")

    run_summary["stage_seconds"] = stage_seconds
    return cast(RunV2PipelineSummary, run_summary)


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
                        # A preflight failure is not a stage failure -- keep the
                        # structured fields present (shape-consistent) but None.
                        "stage": None,
                        "original_error_type": None,
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
                curation_description=curation_description,
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
                    # Surface the failing STAGE and the underlying error type (a
                    # PipelineStageError wraps e.g. an IndexError) so a batch
                    # caller can triage without re-parsing the message. Both are
                    # None for non-stage failures (preflight / zero-unit).
                    "stage": getattr(exc, "stage", None),
                    "original_error_type": getattr(exc, "original_type", None),
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


def run_v2_unit_match(
    plan: "UnitMatchPlan | None" = None,
    *,
    session_group_owner: "str | None" = None,
    session_group_name: "str | None" = None,
    matcher_params_name: str = "unitmatch_default",
    curation_choices: "dict | None" = None,
) -> RunV2UnitMatchSummary:
    """Match units across a SessionGroup's members in one call, then track them.

    Two call forms:

    - ``run_v2_unit_match(plan)`` -- a :class:`UnitMatchPlan` from
      :func:`plan_v2_unit_match` (the recommended plan-then-run path; the plan
      already pinned one curation per member via a curation strategy). A plan
      that could not resolve every member (``plan.ok is False``) raises here.
    - ``run_v2_unit_match(session_group_owner=..., session_group_name=...,
      matcher_params_name=..., curation_choices={...})`` -- the explicit form,
      for power users.

    Chains the cross-session unit-matching stages into one call:
    ``UnitMatchSelection.insert_selection`` -> ``UnitMatch`` (pairwise matches)
    -> ``TrackedUnit`` (biological-unit identity across sessions). Idempotent:
    re-running with the same inputs reuses the existing rows.

    In the explicit form ``curation_choices`` is REQUIRED -- this helper never
    auto-picks a "latest" curation (which would make the match irreproducible
    the moment a source session gains a new curation). Build it from
    :func:`describe_unit_match_choices`, or use the plan form (recommended):
    :func:`plan_v2_unit_match` pins the curations for you by a named curation
    strategy and returns the ``UnitMatchPlan`` you pass here.

    Parameters
    ----------
    plan : UnitMatchPlan, optional
        Reviewable plan returned by :func:`plan_v2_unit_match`. Recommended.
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
        ``session_group_owner`` / ``session_group_name`` / ``matcher_params_name``,
        the ``unit_match_id`` selection PK, ``n_pairs`` (pairwise matches) and
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
    ValueError
        From ``insert_selection`` on a missing/extra member choice, a
        non-existent curation, or a curation that does not belong to its member.
    PipelineStageError
        If ``UnitMatch`` / ``TrackedUnit`` populate fails (names the stage,
        carries the partial summary). A missing optional matcher backend (e.g.
        ``UnitMatchPy``) surfaces here with the backend's install hint.
    """
    from spyglass.spikesorting.v2._unit_match_planning import UnitMatchPlan
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    # Accept a UnitMatchPlan (from plan_v2_unit_match) OR the explicit keyword
    # form. Unwrap a plan into the explicit inputs; a plan that could not pin a
    # curation for every member (not ok) raises here rather than running a
    # partial match.
    if plan is not None:
        if not isinstance(plan, UnitMatchPlan):
            raise PipelineInputError(
                "run_v2_unit_match: plan must be a UnitMatchPlan from "
                "plan_v2_unit_match, or omit plan and pass "
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
        if not plan.ok:
            raise PipelineInputError(
                "run_v2_unit_match: the plan could not pin a curation for "
                "every member (curation_strategy="
                f"{plan.curation_strategy!r}):\n  - "
                + "\n  - ".join(plan.errors)
            )
        session_group_owner = plan.session_group_owner
        session_group_name = plan.session_group_name
        matcher_params_name = plan.matcher_params_name
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
    if curation_choices is None:
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

    run_summary: dict[str, Any] = {
        "session_group_owner": session_group_owner,
        "session_group_name": session_group_name,
        "matcher_params_name": matcher_params_name,
    }
    stage_seconds: dict[str, float] = {}

    # insert_selection pins one curation per member and validates coverage /
    # per-member ownership / geometry BEFORE any populate (a wrong-member or
    # missing choice raises here, not deep in the matcher).
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
    # explicit choices so the summary carries it even on a reused selection.
    warnings: list[str] = []
    choices_by_member = {
        int(idx): (choice["sorting_id"], int(choice["curation_id"]))
        for idx, choice in curation_choices.items()
    }
    divergent = UnitMatchSelection._divergent_electrode_space_members(
        choices_by_member
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
        lambda: UnitMatch.populate(selection, reserve_jobs=False),
        run_summary,
    )
    run_summary["n_pairs"] = int((UnitMatch & selection).fetch1("n_pairs"))

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
        lambda: TrackedUnit.populate(selection, reserve_jobs=False),
        run_summary,
    )
    run_summary["n_tracked_units"] = len(TrackedUnit & selection)

    run_summary["stage_seconds"] = stage_seconds
    return cast(RunV2UnitMatchSummary, run_summary)


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
        # time -- surfacing it here would offer an un-pickable choice.
        recording_ids = list(
            (RecordingSelection & member_id).fetch("recording_id")
        )
        choices: list[dict] = []
        if recording_ids:
            sorting_ids = list(
                (
                    SortingSelection.RecordingSource
                    & [{"recording_id": rid} for rid in recording_ids]
                ).fetch("sorting_id")
            )
            if sorting_ids:
                choices = (
                    CurationV2 & [{"sorting_id": sid} for sid in sorting_ids]
                ).fetch(
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
