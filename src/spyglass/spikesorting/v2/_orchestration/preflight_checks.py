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

    Lets the concat preflight reuse the ``_check_*`` helpers. The message is
    the failing check's fix, prefixed with ``"{caller}: "`` when given.
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
    # ``RecordingSelection`` references ``Raw``, which a partial ingest can miss.
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
    # The recording build also reads 'raw data valid times' for sample bounds.
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
    # A master row can exist with zero members after a partial delete.
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
        # Coincident contacts cannot produce a probe.
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

    Re-checks the params against the installed SI wrapper, which may differ
    from the one present at insert.

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

    The recipe is resolved from the preprocessing recipe. The name is returned
    even when ``sort_checks`` is False, for the resource notes.
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

    The preset's metric and rule rows plus the metric analyzer recipe.
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

    The sorter row's snippet window is in samples, so a rate mismatch
    silently mistunes it. Skipped for rate-agnostic presets and when ``Raw``
    is missing.
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

    Checks that the recipe resolves, preprocessing filters, each group's
    registered geometry supports the estimation recipe, and an ``"apply"``
    sort does not run the sorter's own motion correction.

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

    The mode-independent half of ``preflight_v2_pipeline``, used by
    :func:`assert_concat_preflight`. Raises on the first failing check.
    ``sort_checks=False`` stops after the preprocessing and artifact rows.

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
    # No preset name; the container advisory is dropped.
    _check_sorter_execution(
        check,
        [],
        bundle,
        None,
        sorter_row,
    )
    return sorter_row
