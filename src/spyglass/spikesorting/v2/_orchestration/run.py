"""Run one sort group or concatenated session group through the v2 pipeline.

Public entry points remain available through ``v2.pipeline``. Source building,
stage execution, motion, session batches, and matching have separate owners;
this module composes those operations for one sorting request."""

from __future__ import annotations

import uuid
from typing import Any

from spyglass.spikesorting.v2._orchestration.preflight import (
    resolve_motion_recipe,
    resolve_preset_sort_config,
)
from spyglass.spikesorting.v2._core.recipe_catalog import (
    DEFAULT_PIPELINE_PRESET,
)
from spyglass.spikesorting.v2._orchestration.types import MotionMode
from spyglass.spikesorting.v2._orchestration.motion import (
    _run_motion_correction,
    _run_motion_estimate,
)
from spyglass.spikesorting.v2._orchestration.source import (
    _build_run_source,
    _run_preflight,
    _validate_run_request,
)
from spyglass.spikesorting.v2._orchestration.stages import (
    _assert_figpack_installed,
    _run_auto_curation_stage,
    _run_figpack_stage,
    _run_member_curation_stage,
    _run_root_curation_stage,
    _run_sort_group_keys,
    _run_sorting_stage,
)
from spyglass.spikesorting.v2.curation_api import RunResult


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

    With ``preflight=True`` (the default) this call verifies those
    prerequisites in ~1 s before any populate and raises ``PreflightError``
    with the exact fix if one is missing; call ``preflight_v2_pipeline(...)``
    directly to inspect the report without running.

    Parameters
    ----------
    nwb_file_name
        Session whose data will be sorted. The session must already be
        ingested via ``insert_sessions``.
    sort_group_id
        ID of an existing ``SortGroupV2`` row for this session (see
        Prerequisites; the orchestrator does not create sort groups).
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
        sorts the result. Any preset runs in either mode; the mode is set by
        which inputs are given. The artifact recipe is applied independently
        to each member.
    pipeline_preset
        Pipeline-preset name from ``_PIPELINE_PRESETS``. The default is
        ``franklab_probe_hippocampus_30khz_ms4_2026_06`` (MountainSort4),
        which runs under the v2 ``numpy>=2`` baseline out of the box; it is the
        probe-labeled twin of the tetrode-labeled MS4 preset (both resolve to the
        same parameter rows -- ``probe_type`` is informational).

        MountainSort4 is the scientifically-preferred polymer-probe recipe.
        Run it natively in the standard v2 ``numpy>=2`` environment via
        ``franklab_probe_hippocampus_30khz_ms4_2026_06``, or use the
        ``franklab_probe_hippocampus_30khz_ms4_singularity_2026_06`` preset
        with Singularity/Apptainer. Preflight fails
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
        remains ``None`` for concat. All stay ``None`` on a root-only run.
        Automatic labels are suggestions written as labels, not approval,
        and the child still holds EVERY unit -- select the
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
        sort. Given with another mode, or not a UUID, it raises
        ``PipelineInputError`` before any database access.

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
        session-timeline rows. There is deliberately no bare ``merge_id``.
        A zero-unit single-session sort yields an empty (but real) root
        curation/merge row. A concat run leaves only its unsafe
        synthetic-timeline merge IDs ``None``; its member IDs are
        session-safe.

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
        compute stages are wrapped; an error from the cheap ``insert_selection``
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
    # imports what it uses). The auto-curation (``metric_curation``) and
    # FigPack (``figpack_curation``) schema modules load later -- at
    # preflight or in their own stage -- and only when those options are on.
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
    from spyglass.spikesorting.v2._orchestration.preflight import (
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

    # The FigPack view is of the ROOT curation: FigPack publishes raw-namespace
    # curations only (an auto-curated child is in the curation_evaluation
    # namespace).
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
