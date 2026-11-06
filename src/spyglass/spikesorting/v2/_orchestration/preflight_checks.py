"""Per-stage prerequisite checks for ``preflight_v2_pipeline``.

The concat preflight runs the same checks through ``_raising_check``: the
preset rows via :func:`assert_preset_compute_rows`, and the per-session,
sort-group, sampling-rate and motion checks once per member.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from spyglass.spikesorting.v2._orchestration.preflight_geometry import (
    sort_group_geometry_problem,
)
from spyglass.spikesorting.v2._orchestration.preflight_motion import (
    MotionRecipe,
    motion_geometry_problem,
    resolve_motion_recipe,
    sorter_motion_correction_problem,
)
from spyglass.spikesorting.v2._orchestration.preflight_runtime import (
    _check_sorter_execution,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# Relative tolerance for a recording's sampling rate against the rate a
# preset is tuned for; 0.5% absorbs float drift in an estimated rate.
_SAMPLING_RATE_TOLERANCE = 0.005


def _sampling_rate_matches(actual_rate: float, bundle) -> bool:
    """Whether ``actual_rate`` is within tolerance of the preset's rate.

    Parameters
    ----------
    actual_rate : float
        The recording's sampling rate, in Hz.
    bundle : _PipelinePreset
        The resolved preset; its ``sampling_rate_hz`` must not be ``None``.

    Returns
    -------
    bool
        True when the rates differ by at most ``_SAMPLING_RATE_TOLERANCE``
        of ``bundle.sampling_rate_hz``.
    """
    return (
        abs(actual_rate - bundle.sampling_rate_hz)
        <= _SAMPLING_RATE_TOLERANCE * bundle.sampling_rate_hz
    )


def _raising_check(
    caller: "str | None" = None,
) -> "Callable[[str, Any, str], bool]":
    """A check callback that raises ``PreflightError`` on the first failure.

    Lets the raising concat preflight run the same ``_check_*`` helpers as the
    report-building :func:`preflight_v2_pipeline`. The failing check's fix is
    the message, prefixed with ``"{caller}: "`` when ``caller`` is given.
    """
    from spyglass.spikesorting.v2.exceptions import PreflightError

    def _check(name: str, ok, fix: str) -> bool:
        if not bool(ok):
            raise PreflightError(fix if caller is None else f"{caller}: {fix}")
        return True

    return _check


def _check_session_rows(
    check: "Callable[[str, Any, str], bool]",
    nwb_file_name: str,
    interval_list_name: str,
    team_name: str,
) -> None:
    """Check the session, Raw, interval and team rows a recording build reads.

    Parameters
    ----------
    check : Callable[[str, Any, str], bool]
        The report's check-recording closure.
    nwb_file_name, interval_list_name, team_name : str
        As in :func:`preflight_v2_pipeline`.
    """
    from spyglass.common import IntervalList, LabTeam, Raw, Session

    check(
        "session_exists",
        Session & {"nwb_file_name": nwb_file_name},
        f"session {nwb_file_name!r} is not ingested. Ingest it with "
        "insert_sessions(...) first.",
    )
    # ``RecordingSelection`` FKs ``Raw`` (not ``Session``), so a session whose
    # ``Raw`` row is missing (e.g. a partial ingestion) would pass
    # ``session_exists`` yet fail the recording insert with an opaque
    # foreign-key error. Check ``Raw`` explicitly so preflight stays honest.
    check(
        "raw_exists",
        Raw & {"nwb_file_name": nwb_file_name},
        f"Raw electrical-series row for {nwb_file_name!r} is missing (the "
        "session is ingested but its Raw data is not). Re-run ingestion "
        "(e.g. populate_all_common / insert_sessions) so Raw is populated.",
    )
    check(
        "interval_exists",
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        },
        f"interval_list_name {interval_list_name!r} not found for "
        f"{nwb_file_name!r}. A full-session sort typically uses "
        "'raw data valid times'.",
    )
    # The recording build reads BOTH the sort interval (above) and the raw
    # 'raw data valid times' interval -- for the raw sample bounds -- even when
    # the sort interval differs. A partial ingest can have the sort interval but
    # miss 'raw data valid times', which would otherwise surface late as a bare
    # fetch1 error deep in Recording.make_fetch. Check it up front.
    check(
        "raw_valid_times_exists",
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        },
        f"IntervalList 'raw data valid times' not found for {nwb_file_name!r}; "
        "the recording build reads it for the raw sample bounds. A partial "
        "ingest can miss it -- re-run ingestion (populate_all_common / "
        "insert_sessions).",
    )
    check(
        "team_exists",
        LabTeam & {"team_name": team_name},
        f"LabTeam {team_name!r} does not exist. Create it with "
        "LabTeam.insert1({'team_name': ..., 'team_description': ...}).",
    )


def _check_sort_group(
    check: "Callable[[str, Any, str], bool]",
    nwb_file_name: str,
    sort_group_id: int,
) -> bool:
    """Check the sort group exists, has electrodes and has distinct contacts.

    Returns
    -------
    bool
        Whether the ``SortGroupV2`` master row exists.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    sort_group_exists = check(
        "sort_group_exists",
        SortGroupV2
        & {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id},
        f"SortGroupV2 sort_group_id={sort_group_id} not found for "
        f"{nwb_file_name!r}. Create sort groups first with "
        "SortGroupV2.set_group_by_shank(nwb_file_name=...).",
    )
    # A SortGroupV2 master row can exist with ZERO electrode members (created
    # then partially deleted, or a shank that resolved to an empty group);
    # Recording.populate then raises "has zero electrodes" minutes into the
    # run. Checking the master alone is a false-green. Only run this when the
    # master exists, to avoid a confusing second failure when it is absent.
    if sort_group_exists:
        check(
            "sort_group_has_electrodes",
            SortGroupV2.SortGroupElectrode
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
            },
            f"SortGroupV2 sort_group_id={sort_group_id} for {nwb_file_name!r} "
            "has zero electrode members; Recording.populate would raise 'has "
            "zero electrodes'. Recreate it with "
            "SortGroupV2.set_group_by_shank(nwb_file_name=...).",
        )
        # Geometry the sort would actually see: a group whose contacts
        # coincide cannot produce a probe, and the failure otherwise lands
        # minutes into Recording.populate (or later, at the analyzer build).
        geometry_problem = sort_group_geometry_problem(
            nwb_file_name, sort_group_id
        )
        check(
            "sort_group_geometry_distinct",
            geometry_problem is None,
            geometry_problem or "",
        )
    return sort_group_exists


def _check_source_param_rows(
    check: "Callable[[str, Any, str], bool]", bundle
) -> bool:
    """Check the preset's preprocessing and artifact-detection Lookup rows.

    Returns
    -------
    bool
        Whether the ``PreprocessingParameters`` row exists.
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    preprocessing_params_exist = check(
        "preprocessing_params_exist",
        PreprocessingParameters
        & {"preprocessing_params_name": bundle.preprocessing_params_name},
        f"PreprocessingParameters row {bundle.preprocessing_params_name!r}"
        " is missing. Run initialize_v2_defaults().",
    )
    # An explicit no-mask preset has no artifact parameter row to require.
    if bundle.artifact_detection_params_name is not None:
        check(
            "artifact_detection_params_exist",
            ArtifactDetectionParameters
            & {
                "artifact_detection_params_name": bundle.artifact_detection_params_name
            },
            "ArtifactDetectionParameters row "
            f"{bundle.artifact_detection_params_name!r}"
            " is missing. Run initialize_v2_defaults().",
        )
    return preprocessing_params_exist


def _check_sorter_param_rows(
    check: "Callable[[str, Any, str], bool]", bundle, *, sort_checks: bool
) -> tuple:
    """Check the preset's ``SorterParameters`` row and its parameters.

    The row's params are re-checked against the installed SI wrapper's
    parameter vocabulary (a custom row inserted before the wrapper changed, or
    under a different SI, would otherwise fail minutes into the sort).

    Returns
    -------
    sorter_params_exist : bool
        Whether the row exists; False when ``sort_checks`` is False.
    sorter_row : dict or None
        The fetched row, or ``None`` when it was not checked.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    sorter_params_query = SorterParameters & {
        "sorter": bundle.sorter,
        "sorter_params_name": bundle.sorter_params_name,
    }
    sorter_params_exist = sort_checks and check(
        "sorter_params_exist",
        sorter_params_query,
        f"SorterParameters row (sorter={bundle.sorter!r}, "
        f"sorter_params_name={bundle.sorter_params_name!r})"
        " is missing. Run initialize_v2_defaults().",
    )
    sorter_row = None
    if sorter_params_exist:
        from spyglass.spikesorting.v2._params.sorter import (
            validate_sorter_params_against_wrapper,
        )

        sorter_row = sorter_params_query.fetch1()
        try:
            validate_sorter_params_against_wrapper(
                sorter_row["sorter"], sorter_row["params"]
            )
        except ValueError as exc:
            check("sorter_params_valid", False, str(exc))
        else:
            check("sorter_params_valid", True, "")
    return sorter_params_exist, sorter_row


def _check_display_waveform_params(
    check: "Callable[[str, Any, str], bool]", bundle, *, sort_checks: bool
) -> str:
    """Check the display analyzer recipe's row; return the recipe name.

    The display analyzer recipe is region-resolved from the preprocessing
    recipe and FK-required on the Sorting row, so Sorting.make_fetch fails if
    its AnalyzerWaveformParameters row is missing. Gate it here (up front)
    rather than crashing deep in populate, matching the other params checks.
    The name is returned even when ``sort_checks`` is False, for the
    resource notes.
    """
    from spyglass.spikesorting.v2._core.recipe_catalog import (
        waveform_params_for_preprocessing,
    )
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    display_waveform_params_name = waveform_params_for_preprocessing(
        bundle.preprocessing_params_name
    )[0]
    if sort_checks:
        check(
            "analyzer_waveform_params_exist",
            AnalyzerWaveformParameters
            & {"waveform_params_name": display_waveform_params_name},
            "AnalyzerWaveformParameters row "
            f"{display_waveform_params_name!r} (the display analyzer "
            f"recipe for preprocessing {bundle.preprocessing_params_name!r})"
            " is missing. Run initialize_v2_defaults().",
        )
    return display_waveform_params_name


def _check_auto_curation_rows(
    check: "Callable[[str, Any, str], bool]", bundle
) -> None:
    """Check the rows ``auto_curate=True`` scores the root curation with.

    CurationEvaluation scores the root curation with the preset's metric +
    auto-curation rule rows on the whitened (metric) analyzer recipe, so a
    missing one of those would otherwise fail only after the upstream compute.
    The metric waveform row is the [1] element of the same source-resolved
    (display, metric) pair.
    """
    from spyglass.spikesorting.v2._core.recipe_catalog import (
        waveform_params_for_preprocessing,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    check(
        "metric_params_exist",
        QualityMetricParameters
        & {"metric_params_name": bundle.metric_params_name},
        f"QualityMetricParameters row {bundle.metric_params_name!r} (the "
        "auto-curation metric set)"
        " is missing. Run initialize_v2_defaults().",
    )
    check(
        "auto_curation_rules_exist",
        AutoCurationRules
        & {"auto_curation_rules_name": bundle.auto_curation_rules_name},
        f"AutoCurationRules row {bundle.auto_curation_rules_name!r} (the "
        "auto-curation rule set)"
        " is missing. Run initialize_v2_defaults().",
    )
    metric_waveform_params_name = waveform_params_for_preprocessing(
        bundle.preprocessing_params_name
    )[1]
    check(
        "metric_waveform_params_exist",
        AnalyzerWaveformParameters
        & {"waveform_params_name": metric_waveform_params_name},
        f"AnalyzerWaveformParameters row {metric_waveform_params_name!r} "
        "(the whitened metric analyzer recipe auto-curation scores on)"
        " is missing. Run initialize_v2_defaults().",
    )


def _check_sampling_rate(
    check: "Callable[[str, Any, str], bool]",
    bundle,
    nwb_file_name: str,
    pipeline_preset: "str | None",
) -> None:
    """Check the recording samples at the rate the preset is tuned for.

    The MS4/MS5 snippet window (clip_size / detect_interval on the rate-keyed
    sorter row) assumes a specific acquisition rate, so a 30 kHz preset on a
    20 kHz recording (or the reverse) silently sorts with a mistuned window.
    The clusterless preset is rate-agnostic (sampling_rate_hz is None) and is
    skipped; the check is also skipped if Raw is not ingested yet (raw_exists
    already reports that, so this would only add a confusing second failure).
    A ``pipeline_preset`` of ``None`` (a caller holding only the resolved
    bundle, e.g. the concat preflight) names "the preset" instead.
    """
    from spyglass.common import Raw

    if bundle.sampling_rate_hz is not None:
        raw = Raw & {"nwb_file_name": nwb_file_name}
        if raw:
            actual_rate = float(raw.fetch1("sampling_rate"))
            preset_label = (
                "the preset"
                if pipeline_preset is None
                else f"pipeline_preset {pipeline_preset!r}"
            )
            check(
                "sampling_rate_matches",
                _sampling_rate_matches(actual_rate, bundle),
                f"recording {nwb_file_name!r} samples at {actual_rate:g} Hz "
                f"but {preset_label} is tuned for "
                f"{bundle.sampling_rate_hz} Hz: the rate-keyed sorter row "
                f"{bundle.sorter_params_name!r} holds its clip_size / "
                "detect_interval snippet window at that rate. Pick the "
                "rate-matched preset (call describe_pipeline_presets() and "
                "match sampling_rate_hz to the recording).",
            )


def _check_motion_stage(
    check: "Callable[[str, Any, str], bool]",
    bundle,
    *,
    sort_groups: "list[tuple[Callable[[str, Any, str], bool], str, int]]",
    motion_mode,
    motion_correction_params_name: str,
    preprocessing_params_exist: bool,
    sorter_row: "dict | None",
) -> "MotionRecipe | None":
    """Check a motion stage (estimate / apply) can run; return its recipe.

    The recipe must exist and resolve, the preset's preprocessing recipe must
    filter (motion is estimated on filtered, unwhitened traces;
    MotionEstimateSelection.insert_selection refuses the same source), each
    sort group's effective geometry must support the estimation recipe
    (checked here on the registered positions, before any populate; the
    estimator re-checks the actual recording), and a sort of a corrected
    recording must not run the sorter's own motion correction.

    Parameters
    ----------
    check : Callable[[str, Any, str], bool]
        The check callback for the run-level checks.
    bundle : _PipelinePreset
        The resolved preset.
    sort_groups : list of (check, nwb_file_name, sort_group_id)
        The sort groups whose geometry is checked, each through its own
        callback so a concat run can name the member; empty skips the
        geometry check.
    motion_mode, motion_correction_params_name
        The motion request.
    preprocessing_params_exist : bool
        Whether the preset's ``PreprocessingParameters`` row exists; the
        filter check is skipped otherwise.
    sorter_row : dict or None
        The preset's ``SorterParameters`` row; ``None`` (not checked or
        missing) skips the sorter's motion-correction check.

    Returns
    -------
    MotionRecipe or None
        The resolved recipe, or ``None`` when it does not resolve.
    """
    from spyglass.spikesorting.v2.motion import (
        preprocessing_filter_problem,
    )

    motion_recipe = None
    if preprocessing_params_exist:
        unfiltered = preprocessing_filter_problem(
            bundle.preprocessing_params_name
        )
        check(
            "motion_source_filtered",
            unfiltered is None,
            f"motion_mode={motion_mode!r}: {unfiltered}",
        )
    try:
        motion_recipe = resolve_motion_recipe(motion_correction_params_name)
    except ValueError as exc:
        check("motion_recipe_exists", False, str(exc))
    else:
        check("motion_recipe_exists", True, "")
        for group_check, nwb_file_name, sort_group_id in sort_groups:
            motion_geometry = motion_geometry_problem(
                nwb_file_name,
                sort_group_id,
                motion_recipe.resolved_estimation,
            )
            group_check(
                "motion_geometry_supported",
                motion_geometry is None,
                motion_geometry or "",
            )
    if motion_mode == "apply" and sorter_row is not None:
        sorter_motion = sorter_motion_correction_problem(
            bundle.sorter, sorter_row["params"], bundle.sorter_params_name
        )
        check(
            "sorter_motion_correction_off",
            sorter_motion is None,
            sorter_motion or "",
        )
    return motion_recipe


def assert_preset_compute_rows(
    bundle, *, caller: str = "run_v2_concat_pipeline", sort_checks: bool = True
) -> "dict | None":
    """Raise ``PreflightError`` if a preset's compute-time rows / sorter binary
    are missing.

    The mode-independent half of ``preflight_v2_pipeline``: the preset's
    preprocessing / (optional) artifact / sorter / display-analyzer-waveform
    Lookup rows plus the sorter binary/runtime. :func:`assert_concat_preflight`
    (run by ``run_v2_concat_pipeline`` / ``estimate_concat_motion``) calls
    this, including its member artifact recipe, so a concat run fails fast on a
    missing row instead of deep in the member/concat populate, matching the
    single-session preflight. The checks run ``preflight_v2_pipeline``'s own
    ``_check_*`` helpers through :func:`_raising_check`, so the first failing
    check raises with its fix prefixed by ``caller``. ``sort_checks=False``
    stops after the preprocessing and artifact rows (a caller that builds the
    source without sorting it).

    Returns
    -------
    dict or None
        The preset's ``SorterParameters`` row, or ``None`` when
        ``sort_checks`` is False.
    """
    check = _raising_check(caller)
    _check_source_param_rows(check, bundle)
    if not sort_checks:
        return None
    sorter_row = _check_sorter_param_rows(check, bundle, sort_checks=True)[1]
    _check_display_waveform_params(check, bundle, sort_checks=True)
    # The sorter binary/runtime on the row's execution backend, through the
    # single-session check (the row exists: the check above raised otherwise).
    # The bundle carries no preset name, so the messages name none, and a
    # runnable container backend's advisory is dropped.
    _check_sorter_execution(
        check,
        [],
        bundle,
        None,
        sorter_row,
    )
    return sorter_row
