"""Read-only configuration checks run before any pipeline populate.

Holds ``preflight_v2_pipeline`` / ``preflight_v2_pipeline_session`` and their
``Preflight*`` report types, the raising ``assert_*`` guards that concat-mode
preflight runs (``assert_concat_preflight`` / ``assert_preset_compute_rows``),
motion-recipe and geometry checks, and ``describe_scientific_setup``.
``pipeline.py`` re-exports the preflight entry points and report types.
At import time this module depends only on ``_pipeline_presets``,
``_pipeline_types``, and ``_recipe_catalog``; table modules are imported
inside the functions that query them. ``_pipeline_run`` imports this module.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pprint import pformat
from typing import TYPE_CHECKING, Any, NamedTuple, get_args

from spyglass.spikesorting.v2._pipeline_presets import (
    _PIPELINE_PRESETS,
    _unknown_pipeline_preset_message,
)
from spyglass.spikesorting.v2._pipeline_types import MotionMode
from spyglass.spikesorting.v2._recipe_catalog import DEFAULT_PIPELINE_PRESET

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd


# SpikeInterface's ``installed_sorters()`` reports a sorter as installed when
# its thin wrapper imports, but some sorters call a separate algorithm backend
# at run time that the wrapper does NOT import -- so the check over-reports. The
# live example is ``mountainsort4``: its wrapper imports (so it appears in
# ``installed_sorters()``), but the actual algorithm package ``ml_ms4alg`` is a
# numpy<2-era build that no longer installs under the v2 ``numpy>=2`` baseline.
# Map each such sorter to the backend module(s) preflight must additionally
# verify, so a green ``sorter_installed`` cannot precede a sort-time
# ``ModuleNotFoundError``.
_SORTER_RUNTIME_BACKENDS: dict[str, tuple[str, ...]] = {
    "mountainsort4": ("ml_ms4alg",),
}

# Relative tolerance for a recording's sampling rate against the rate a
# preset is tuned for; 0.5% absorbs float drift in an estimated rate.
_SAMPLING_RATE_TOLERANCE = 0.005


def motion_request_problem(
    motion_mode, motion_correction_params_name, motion_estimate_id=None
) -> "str | None":
    """Say why a motion mode and recipe name cannot be run together.

    DB-free: it checks the request's shape, not that the named
    ``MotionCorrectionParameters`` row or motion estimate exists.

    Parameters
    ----------
    motion_mode : str
        One of ``"off"``, ``"estimate"``, ``"apply"``.
    motion_correction_params_name : str or None
        The ``MotionCorrectionParameters`` row; required iff ``motion_mode``
        is not ``"off"``.
    motion_estimate_id : uuid.UUID or str, optional
        A saved motion estimate to apply; only valid with ``"apply"``.

    Returns
    -------
    str or None
        The operator-facing problem, or ``None`` for a valid request.
    """
    problem = _motion_mode_problem(motion_mode, motion_correction_params_name)
    if problem is not None or motion_estimate_id is None:
        return problem
    if motion_mode != "apply":
        return (
            f"motion_estimate_id={motion_estimate_id!r} was given with "
            f"motion_mode={motion_mode!r}; only motion_mode='apply' applies "
            "a saved estimate. Pass motion_mode='apply', or drop the id."
        )
    try:
        uuid.UUID(str(motion_estimate_id))
    except ValueError:
        return (
            f"motion_estimate_id={motion_estimate_id!r} is not a UUID; pass "
            "the motion_estimate_id of a saved MotionEstimate."
        )
    return None


def _motion_mode_problem(
    motion_mode, motion_correction_params_name
) -> "str | None":
    """The mode / recipe-name half of :func:`motion_request_problem`."""
    modes = get_args(MotionMode)
    if motion_mode not in modes:
        return (
            f"unknown motion_mode {motion_mode!r}; expected one of "
            f"{list(modes)}."
        )
    if motion_mode == "off":
        if motion_correction_params_name is not None:
            return (
                "motion_correction_params_name="
                f"{motion_correction_params_name!r} was given with "
                "motion_mode='off', which runs no motion stage. Pass "
                "motion_mode='estimate' or 'apply' to use the recipe, or drop "
                "the name."
            )
        return None
    if motion_correction_params_name is None:
        return (
            f"motion_mode={motion_mode!r} requires "
            "motion_correction_params_name, a MotionCorrectionParameters row "
            "(e.g. 'dredge_fast_v1'; see MotionCorrectionParameters())."
        )
    return None


def _docker_runtime_available() -> tuple[bool, str]:
    """Return ``(ok, detail)`` for the Docker container runtime.

    A container-backed (``backend="docker"``) preset needs both the Docker
    engine and the Python ``docker`` package -- the two things SpikeInterface's
    ``run_sorter`` itself checks before dispatching to a Docker container.
    Defined as a module-level function so tests can monkeypatch the runtime
    probe without a real Docker install.
    """
    from spikeinterface.sorters.runsorter import (
        has_docker,
        has_docker_python,
    )

    if not has_docker():
        return False, "the Docker engine (`docker` CLI) was not found"
    if not has_docker_python():
        return (
            False,
            "the Python `docker` package is not installed "
            "(`pip install docker`)",
        )
    return True, "Docker engine + Python `docker` package available"


def _singularity_runtime_available() -> tuple[bool, str]:
    """Return ``(ok, detail)`` for the Singularity/Apptainer container runtime.

    A container-backed (``backend="singularity"``) preset needs both Singularity
    (or Apptainer) and the Python ``spython`` package -- the two things
    SpikeInterface's ``run_sorter`` checks before dispatching to a Singularity
    container. Defined as a module-level function so tests can monkeypatch the
    runtime probe without a real Singularity install.
    """
    from spikeinterface.sorters.runsorter import (
        has_singularity,
        has_spython,
    )

    if not has_singularity():
        return False, "Singularity/Apptainer was not found"
    if not has_spython():
        return (
            False,
            "the Python `spython` package is not installed "
            "(`pip install spython`)",
        )
    return True, "Singularity + Python `spython` package available"


def _container_runtime_available(execution_backend: str) -> tuple[bool, str]:
    """Return ``(ok, detail)`` for a container execution backend's runtime.

    ``"docker"`` probes :func:`_docker_runtime_available`; any other container
    backend probes :func:`_singularity_runtime_available`.
    """
    if execution_backend == "docker":
        return _docker_runtime_available()
    return _singularity_runtime_available()


def _check_local_sorter_runtime(bundle, sis, non_si_sorters, check) -> None:
    """Run the LOCAL-execution sorter checks (installed + runtime backend).

    Shared by :func:`_check_sorter_execution`, whose execution-backend
    dispatch is a flat three-arm choice (MATLAB-local error / local checks /
    container runtime), and by :func:`assert_preset_compute_rows`.

    Parameters
    ----------
    bundle : _PipelinePreset
        The resolved preset whose ``sorter`` is being checked.
    sis : module
        ``spikeinterface.sorters``, imported lazily by each caller.
    non_si_sorters : Container[str]
        ``SorterParameters._NON_SI_SORTERS`` -- never gated on an SI binary.
    check : Callable[[str, Any, str], bool]
        The report's check-recording closure.
    """
    # sorter_installed. Reuse the SAME strict gate insert_default uses: the
    # internal clusterless_thresholder (_NON_SI_SORTERS) is never an SI binary
    # and is always available; otherwise the sorter must be in
    # installed_sorters(), distinguishing "known but not installed" from
    # "misspelled / unknown" for the fix message.
    sorter_installed_ok = (
        bundle.sorter in non_si_sorters
        or bundle.sorter in set(sis.installed_sorters())
    )
    if sorter_installed_ok:
        check("sorter_installed", True, "")
    elif bundle.sorter in set(sis.available_sorters()):
        check(
            "sorter_installed",
            False,
            f"sorter {bundle.sorter!r} is a known SpikeInterface sorter but its "
            "binary/runtime is not installed here "
            "(spikeinterface.sorters.installed_sorters()). Install it, or pick a "
            "preset whose sorter is installed. (Or use a containerized execution "
            "preset, which runs the sorter runtime inside a container image "
            "instead.)",
        )
    else:
        check(
            "sorter_installed",
            False,
            f"sorter {bundle.sorter!r} is not a known SpikeInterface sorter "
            "(spikeinterface.sorters.available_sorters()) -- check the spelling "
            "or the preset.",
        )

    # sorter_runtime_available. installed_sorters() only checks that the SI
    # wrapper imports; for sorters that call a SEPARATE algorithm backend at run
    # time (see _SORTER_RUNTIME_BACKENDS) actually import that backend, so a
    # green sorter_installed cannot precede a sort-time failure -- whether the
    # backend is absent OR present-but-broken (e.g. a numpy<2-era ml_ms4alg
    # under the numpy>=2 baseline raising at import). Only runs when
    # sorter_installed passed: if the wrapper itself is missing, a second
    # "backend missing" failure would be contradictory.
    backend_modules = _SORTER_RUNTIME_BACKENDS.get(bundle.sorter, ())
    if sorter_installed_ok and backend_modules:
        import importlib

        broken_backends = []
        for mod in backend_modules:
            try:
                importlib.import_module(mod)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - any import failure disqualifies
                broken_backends.append(f"{mod} ({type(exc).__name__}: {exc})")
        check(
            "sorter_runtime_available",
            not broken_backends,
            f"sorter {bundle.sorter!r} is listed as installed but its runtime "
            "backend(s) cannot be imported, so the sort would crash: "
            f"{'; '.join(broken_backends)}. Install/repair the backend "
            "(mountainsort4 needs ml_ms4alg, which requires numpy<2), or pick a "
            "preset whose sorter runs in this environment -- e.g. a MountainSort5 "
            "preset, or a containerized MountainSort4 preset whose runtime lives "
            "in the container.",
        )


@dataclass(frozen=True)
class PreflightCheck:
    """One preflight check outcome: name, pass/fail, and the fix on fail."""

    name: str  # e.g. "session_exists", "sorter_installed"
    ok: bool
    fix: str  # empty when ok; the actionable fix when not ok


@dataclass(frozen=True)
class PreflightReport:
    """Result of ``preflight_v2_pipeline``: a pre-populate config check.

    Truthy when the configuration is runnable (``ok is True``), so a
    notebook can ``if not preflight_v2_pipeline(...): ...``. ``errors``
    lists each blocking problem with its fix; ``warnings`` holds
    non-blocking advisories; ``expected_ids`` carries the deterministic
    selection PKs the run would produce.

    Attributes
    ----------
    ok
        True when no blocking problem was found (``errors`` is empty).
    errors
        Blocking-problem messages; non-empty iff ``ok`` is False.
    warnings
        Non-blocking advisories (e.g. ``artifact_detection_params_name="none"``).
    resolved_pipeline_preset
        The pipeline-preset name that was checked.
    expected_ids
        The selection PKs a subsequent ``run_v2_pipeline`` would produce, each
        annotated with whether its SELECTION row and its COMPUTED output row
        already exist, e.g. ``{"recording_id": {"id": UUID(...),
        "exists": False, "computed_exists": False}, ...}``. ``exists`` is the
        selection ``& pk`` restriction (the run would reuse this PK);
        ``computed_exists`` is the computed-table ``& pk`` restriction (the
        populate already ran, so that stage would be a near-zero-cost reuse) --
        distinguishing them shows what work the run would actually do. IDs are
        computed DB-free via ``deterministic_id``. Empty when the preset is
        unknown (the param names needed to derive the IDs are then unavailable).
        For an ``ok`` report each ``id`` equals the PK ``run_v2_pipeline``
        returns. ``curation_id`` is intentionally excluded: it is assigned by
        ``CurationV2.insert_curation``, not content-addressed. With a motion
        mode, ``motion_estimate_id`` (and for ``"apply"``
        ``motion_corrected_recording_id``) entries precede ``sorting_id``;
        the estimate id includes the recording's content hash, so before the
        recording is computed those entries -- and an ``"apply"`` run's
        ``sorting_id`` -- have ``id=None`` and a ``pending`` reason.
    checks
        Per-check detail; every check runs (the report is complete, not
        first-failure-only).
    resource_notes
        Known large allocations and scratch requirements the run implies,
        stated from the tracked recipes (no unit count is known before
        sorting): the per-unit waveform buffer upper bound of the display
        analyzer (``max_spikes_per_unit`` x window x channels x 4 bytes), the
        analyzer cache and sorter scratch locations, and the effective worker /
        chunk settings. Informational; never a blocking check.
    effective_config
        What the sort stage would actually execute, as
        ``EffectiveSortConfig.as_dict()`` (sorter, the kwargs handed to
        SpikeInterface, external-whiten routing, seed, resolved job kwargs,
        execution backend). Resolved by the SAME ``resolve_sort_config`` the
        dispatcher runs, from the preset's ``SorterParameters`` row and the
        ambient job-kwargs layer at preflight time. ``None`` when that row is
        missing (``sorter_params_exist`` then reports the fix).
    """

    ok: bool
    errors: list[str]
    warnings: list[str]
    resolved_pipeline_preset: str
    expected_ids: dict
    checks: list["PreflightCheck"]
    effective_config: "dict | None" = None
    resource_notes: list[str] = field(default_factory=list)
    scientific_config: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Return ``True`` when the configuration is runnable (``ok``)."""
        return self.ok

    def summary(self) -> str:
        """Render the execution plan and actionable findings without DB reads."""
        state = "ready" if self.ok else "blocked"
        return "\n".join(
            [f"Preflight {state}: {self.resolved_pipeline_preset}"]
            + _preflight_details(
                self.errors,
                self.warnings,
                self.expected_ids,
                self.effective_config,
                self.resource_notes,
                self.scientific_config,
            )
        )


@dataclass(frozen=True)
class PreflightSessionReport:
    """Result of ``preflight_v2_pipeline_session``: a whole-session check.

    Aggregates a per-sort-group :class:`PreflightReport` into one object.
    Truthy when *every* target group is runnable (``ok is True``), so a
    notebook can ``if not preflight_v2_pipeline_session(...): ...``.

    Attributes
    ----------
    ok
        True when no target group has a blocking problem (``errors`` empty).
    errors
        Blocking-problem messages across all groups, each prefixed with its
        ``sort_group_id``; non-empty iff ``ok`` is False.
    warnings
        Non-blocking advisories across all groups, each prefixed with its
        ``sort_group_id``.
    resolved_pipeline_preset
        The pipeline-preset name that was checked for every group.
    group_reports
        One plain-dict entry per target sort group, with keys
        ``sort_group_id``, ``ok``, ``errors``, ``warnings``, ``expected_ids``,
        ``checks``, ``effective_config``, and ``resource_notes`` (from the
        underlying :class:`PreflightReport`).
        ``pandas`` is intentionally not imported here; wrap with
        ``pd.DataFrame(report.group_reports)`` in a notebook when useful.
    """

    ok: bool
    errors: list[str]
    warnings: list[str]
    resolved_pipeline_preset: str
    group_reports: list[dict[str, Any]]

    def __bool__(self) -> bool:
        """Return ``True`` when every target group is runnable (``ok``)."""
        return self.ok

    def summary(self) -> str:
        """Render the checked targets and their execution plans."""
        state = "ready" if self.ok else "blocked"
        lines = [
            f"Session preflight {state}: {self.resolved_pipeline_preset}",
            "Target sort groups: "
            + ", ".join(
                str(row["sort_group_id"]) for row in self.group_reports
            ),
        ]
        for row in self.group_reports:
            lines.append(f"Sort group {row['sort_group_id']}:")
            lines.extend(
                _preflight_details(
                    row["errors"],
                    row["warnings"],
                    row["expected_ids"],
                    row["effective_config"],
                    row["resource_notes"],
                    row["scientific_config"],
                )
            )
        return "\n".join(lines)


def _preflight_details(
    errors,
    warnings,
    expected_ids,
    effective_config,
    resources,
    scientific_config=None,
):
    """Format fields shared by single-group and session preflight reports."""
    lines = [f"  ERROR: {message}" for message in errors]
    lines.extend(f"  Warning: {message}" for message in warnings)
    if scientific_config:
        lines.append("  Scientific setup:")
        lines.extend(
            f"    {line}"
            for line in pformat(scientific_config, width=76).splitlines()
        )
    for key, selection in expected_ids.items():
        if selection.get("pending"):
            action = f"compute (id known once {selection['pending']})"
        elif selection["id"] is None:
            action = "skip"
        elif selection["computed_exists"]:
            action = "reuse completed output"
        else:
            action = "compute"
        lines.append(f"  {key.removesuffix('_id')}: {action}")
    if effective_config is not None:
        lines.append("  Effective sorter configuration:")
        lines.extend(
            f"    {line}"
            for line in pformat(effective_config, width=76).splitlines()
        )
    if resources:
        lines.append("  Resources:")
        lines.extend(f"    {note}" for note in resources)
    return lines


#: Stated on every motion description: no recipe has passed probe-specific
#: validation.
MOTION_RECIPE_STATUS = (
    "experimental: no motion recipe is validated for a probe; inspect the "
    "saved estimate before relying on a correction"
)


def _describe_motion(
    motion_mode, motion_recipe: "MotionRecipe | None", *, concat: bool
) -> dict:
    """The motion stage a run executes, for the scientific-setup display.

    Parameters
    ----------
    motion_mode : {"off", "estimate", "apply"}
    motion_recipe : MotionRecipe or None
        The resolved recipe; ``None`` for ``"off"`` or when it could not be
        resolved (preflight then reports why).
    concat : bool
        Whether the source is a concatenation.

    Returns
    -------
    dict
        ``mode``, ``description`` and, with a recipe, ``recipe``,
        ``estimation_recipe``, ``estimation_preset`` (resolved), for
        ``"apply"`` ``interpolation_recipe`` and ``border_mode``, and
        ``status``.
    """
    if motion_mode == "off":
        return {
            "mode": "off",
            "description": (
                "No motion stage: the sort reads the masked, uncorrected "
                "source. DriftEstimate is diagnostic only."
            ),
        }
    source = "the concatenation" if concat else "the recording under its mask"
    description = (
        f"Motion is estimated once from {source} and saved for inspection; "
        "the sort reads the uncorrected source (the same sort as 'off')."
        if motion_mode == "estimate"
        else f"Motion is estimated once from {source}, applied to it, and "
        "the sort reads the motion-corrected recording."
    )
    motion = {"mode": motion_mode, "description": description}
    if motion_recipe is None:
        return motion
    motion.update(
        recipe=motion_recipe.recipe["motion_correction_params_name"],
        estimation_recipe=motion_recipe.recipe["motion_estimation_params_name"],
        estimation_preset=motion_recipe.resolved_estimation["preset"],
    )
    if motion_mode == "apply":
        motion.update(
            interpolation_recipe=motion_recipe.recipe[
                "motion_interpolation_params_name"
            ],
            border_mode=motion_recipe.interpolation_params["border_mode"],
        )
    motion["status"] = MOTION_RECIPE_STATUS
    return motion


def describe_scientific_setup(
    bundle,
    group_keys,
    effective_config=None,
    *,
    manual_excluded_times=None,
    concat: bool = False,
    motion_mode: MotionMode = "off",
    motion_recipe: "MotionRecipe | None" = None,
):
    """Resolve the preprocessing and artifact rows execution uses for display.

    ``concat=True`` (the caller's input mode, not a property of the preset)
    adds where the artifact mask is applied: per member, before
    concatenation. ``motion`` states the run's motion stage: the mode, the
    recipe, the SpikeInterface preset its estimation recipe resolves to, the
    border mode of an applied correction and the recipes' experimental
    status; a Kilosort4 sort adds its own drift-correction ``nblocks``.
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        SortGroupV2,
    )

    preprocessing = (
        PreprocessingParameters
        & {"preprocessing_params_name": bundle.preprocessing_params_name}
    ).fetch("params")
    artifacts = (
        (
            ArtifactDetectionParameters
            & {
                "artifact_detection_params_name": bundle.artifact_detection_params_name
            }
        ).fetch("params")
        if bundle.artifact_detection_params_name is not None
        else []
    )
    result = {
        "preprocessing_recipe": bundle.preprocessing_params_name,
        "preprocessing": (
            dict(preprocessing[0]) if len(preprocessing) else None
        ),
        "references": (SortGroupV2 & group_keys).fetch(
            "nwb_file_name",
            "sort_group_id",
            "reference_mode",
            "reference_electrode_id",
            as_dict=True,
        ),
        "artifact_recipe": bundle.artifact_detection_params_name,
        "artifact_detection": dict(artifacts[0]) if len(artifacts) else None,
        "motion": _describe_motion(motion_mode, motion_recipe, concat=concat),
    }
    if manual_excluded_times:
        result["manual_excluded_times"] = manual_excluded_times
    if concat:
        result["artifact_application"] = (
            "No artifact masking selected."
            if not manual_excluded_times
            and (
                bundle.artifact_detection_params_name is None
                or (len(artifacts) and not artifacts[0].get("detect", True))
            )
            else "Per member before concatenation."
        )
    if bundle.sorter == "kilosort4":
        params = (effective_config or {}).get("si_sorter_params", {})
        result["motion"]["sorter_correction"] = {
            "sorter": "kilosort4",
            "nblocks": params.get("nblocks"),
        }
    return result


def resolve_preset_sort_config(bundle) -> "dict | None":
    """Resolve what a preset's sort stage would execute, or ``None`` if unset.

    Fetches the preset's ``SorterParameters`` row and runs the dispatcher's own
    ``resolve_sort_config`` over it with the ambient job-kwargs layer
    (``utils._resolved_job_kwargs``), returning ``EffectiveSortConfig.as_dict()``.
    Shared by ``preflight_v2_pipeline`` (single-session), the concat preflight
    and ``run_v2_pipeline`` (which records it on the run receipt as
    ``sorter_config``), so preflight, the receipt and execution describe the
    same thing. ``None`` when the row does not exist.
    """
    from spyglass.spikesorting.v2._sorting_dispatch import resolve_sort_config
    from spyglass.spikesorting.v2.sorting import SorterParameters
    from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

    rows = (
        SorterParameters
        & {
            "sorter": bundle.sorter,
            "sorter_params_name": bundle.sorter_params_name,
        }
    ).fetch(as_dict=True)
    if len(rows) != 1:
        return None
    row = rows[0]
    return resolve_sort_config(
        row["sorter"],
        row["params"],
        job_kwargs=_resolved_job_kwargs(row["job_kwargs"]),
        execution_params=row.get("execution_params"),
    ).as_dict()


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


def assert_preset_compute_rows(
    bundle, *, caller: str = "run_v2_pipeline", sort_checks: bool = True
) -> None:
    """Raise ``PreflightError`` if a preset's compute-time rows / sorter binary
    are missing.

    The mode-independent half of ``preflight_v2_pipeline``: the preset's
    preprocessing / (optional) artifact / sorter / display-analyzer-waveform
    Lookup rows plus the sorter binary/runtime. ``run_v2_pipeline``'s concat
    preflight calls this, including its member artifact recipe, so a concat
    run fails fast on a missing row instead of deep
    in the member/concat populate, matching the single-session preflight. The
    param-row checks run ``preflight_v2_pipeline``'s own ``_check_*`` helpers
    through :func:`_raising_check`, so the first failing check raises with its
    fix prefixed by ``caller``. ``sort_checks=False`` stops after the
    preprocessing and artifact rows (a caller that builds the source without
    sorting it).
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._params.sorter import (
        validate_execution_params,
    )
    from spyglass.spikesorting.v2._sorting_dispatch import (
        MATLAB_SORTERS,
        matlab_container_required_message,
    )
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.sorting import SorterParameters

    check = _raising_check(caller)
    _check_source_param_rows(check, bundle)
    if not sort_checks:
        return
    sorter_params_query = _check_sorter_param_rows(
        check, bundle, sort_checks=True
    )[0]
    _check_display_waveform_params(check, bundle, sort_checks=True)

    # Sorter binary/runtime, dispatched on the execution backend like the
    # single-session ``_check_sorter_execution``: a LOCAL backend needs the
    # sorter installed here; a CONTAINER backend needs the container runtime
    # (the sorter runtime lives in the image, so a missing LOCAL install is
    # irrelevant, but a missing container runtime is a blocking failure --
    # preflight never falls back to local execution). The MATLAB and local
    # runtime messages carry no ``caller`` prefix; the container message does
    # and, unlike the single-session one, names no preset.
    execution_params = validate_execution_params(
        sorter_params_query.fetch1("execution_params")
    )
    execution_backend = execution_params["backend"]
    container_image = execution_params["container_image"]
    if execution_backend == "local":
        if bundle.sorter.lower() in MATLAB_SORTERS:
            raise PreflightError(
                matlab_container_required_message(bundle.sorter)
            )
        _check_local_sorter_runtime(
            bundle, sis, SorterParameters._NON_SI_SORTERS, _raising_check()
        )
    else:
        runtime_ok, runtime_detail = _container_runtime_available(
            execution_backend
        )
        if not runtime_ok:
            raise PreflightError(
                f"{caller}: the {execution_backend} execution backend "
                f"(image {container_image!r}) for sorter {bundle.sorter!r} is "
                f"not runnable here: {runtime_detail}. Install the container "
                "runtime and its Python package, or pick a local-execution "
                "preset. Preflight does not fall back to local execution."
            )


def assert_concat_preflight(
    concat_session_group_owner,
    concat_session_group_name,
    bundle,
    *,
    auto_curate: bool = False,
    manual_excluded_times=None,
    motion_mode: MotionMode = "off",
    motion_correction_params_name: "str | None" = None,
    motion_estimate_id=None,
    caller: str = "run_v2_pipeline",
    sort_checks: bool = True,
) -> list[str]:
    """Raise ``PreflightError`` if a concat run's prerequisites are missing.

    Concat counterpart of :func:`preflight_v2_pipeline`: the group, its members,
    and -- because each member is sorted through the same single-session
    ``Recording`` build -- the per-member ``Raw`` / ``'raw data valid times'`` /
    sort-group-electrode / sampling-rate prerequisites, plus the
    ``auto_curate`` metric/rule/metric-waveform rows
    (when opted in), and the compute-time param rows + sorter binary via
    :func:`assert_preset_compute_rows`, including member artifact parameters.
    With a motion mode, also the recipe, that the preset's preprocessing
    recipe (the concatenation's) applies a temporal filter, each member's
    geometry against the estimation recipe, and (for ``"apply"``) the
    sorter's own motion correction, as in :func:`preflight_v2_pipeline`.
    A supplied ``motion_estimate_id`` must be a populated estimate, made with
    the recipe's estimation row, of a concatenation of this session group
    under the preset's preprocessing recipe whose frozen members are the
    member recordings and artifact detections this run would select. ``sort_checks=False`` skips
    the sorter-only checks (the sorter rows and runtime, the display analyzer
    recipe and each member's preset sampling rate), as in
    :func:`preflight_v2_pipeline`. Fails before
    member/concat populate. Returns advisory warnings (including
    explicitly disabled artifact masking) for symmetry with
    :func:`preflight_v2_pipeline`.
    """
    group_key = {
        "session_group_owner": concat_session_group_owner,
        "session_group_name": concat_session_group_name,
    }
    members = _assert_session_group_members(group_key, caller)

    # Each member is sorted through the same single-session Recording build, so
    # mirror preflight_v2_pipeline's per-session prerequisites for EVERY member;
    # SessionGroup.Member's FK set validates only that the master rows exist, not
    # Raw / 'raw data valid times' / a non-empty sort group / a matching rate, so
    # a partially-ingested or empty-sort-group member would otherwise fail deep
    # in the member populate with an opaque error.
    for member in members:
        _assert_concat_member_inputs(
            member, bundle, caller=caller, sort_checks=sort_checks
        )

    # Auto-curation prerequisites (preset-level, same rows as single-session),
    # only when the caller opts into auto_curate -- so a concat auto-curate run
    # fails fast on a missing metric / rule / metric-waveform recipe rather than
    # after the member + concat + sort compute.
    if auto_curate:
        _check_auto_curation_rows(_raising_check(caller), bundle)

    assert_preset_compute_rows(bundle, caller=caller, sort_checks=sort_checks)
    if motion_mode != "off":
        _assert_concat_motion_stage(
            bundle,
            members,
            group_key,
            motion_mode=motion_mode,
            motion_correction_params_name=motion_correction_params_name,
            motion_estimate_id=motion_estimate_id,
            manual_excluded_times=manual_excluded_times,
            caller=caller,
        )
    if (
        bundle.artifact_detection_params_name in (None, "none")
        and not manual_excluded_times
    ):
        return ["No artifact masking selected for the concatenated members."]
    return []


def _assert_session_group_members(group_key: dict, caller: str) -> list[dict]:
    """Raise ``PreflightError`` unless the session group has members.

    Returns
    -------
    list[dict]
        The ``SessionGroup.Member`` rows' ``member_index``,
        ``nwb_file_name``, ``sort_group_id``, ``interval_list_name`` and
        ``team_name``.
    """
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.session_group import SessionGroup

    if not (SessionGroup & group_key):
        raise PreflightError(
            f"{caller}: SessionGroup {group_key} does not exist. "
            "Create it with SessionGroup.create_group(...) first."
        )
    if not (SessionGroup.Member & group_key):
        raise PreflightError(
            f"{caller}: SessionGroup {group_key} has no members."
        )
    return (SessionGroup.Member & group_key).fetch(
        "member_index",
        "nwb_file_name",
        "sort_group_id",
        "interval_list_name",
        "team_name",
        as_dict=True,
    )


def _assert_concat_member_inputs(
    member: dict, bundle, *, caller: str, sort_checks: bool
) -> None:
    """Raise ``PreflightError`` if a member cannot be built as a recording.

    Checks the member's ``Raw`` row, its ``'raw data valid times'`` interval,
    its sort-group electrodes and, with ``sort_checks``, that it samples at
    the preset's rate.
    """
    from spyglass.common import IntervalList, Raw
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb = member["nwb_file_name"]
    sort_group_id = int(member["sort_group_id"])
    tag = (
        f"concat member {member['member_index']} "
        f"({nwb!r}, sort_group_id={sort_group_id})"
    )
    if not (Raw & {"nwb_file_name": nwb}):
        raise PreflightError(
            f"{caller}: {tag} has no Raw electrical-series row (the "
            "session is ingested but its Raw data is not). Re-run ingestion "
            "(populate_all_common / insert_sessions)."
        )
    if not (
        IntervalList
        & {
            "nwb_file_name": nwb,
            "interval_list_name": "raw data valid times",
        }
    ):
        raise PreflightError(
            f"{caller}: {tag} is missing IntervalList 'raw data "
            "valid times', which the recording build reads for the raw "
            "sample bounds. Re-run ingestion."
        )
    if not (
        SortGroupV2.SortGroupElectrode
        & {"nwb_file_name": nwb, "sort_group_id": sort_group_id}
    ):
        raise PreflightError(
            f"{caller}: {tag} SortGroupV2 has zero electrode "
            "members; Recording.populate would raise 'has zero electrodes'. "
            "Recreate it with SortGroupV2.set_group_by_shank(nwb_file_name="
            "...)."
        )
    if sort_checks and bundle.sampling_rate_hz is not None:
        actual_rate = float(
            (Raw & {"nwb_file_name": nwb}).fetch1("sampling_rate")
        )
        if (
            abs(actual_rate - bundle.sampling_rate_hz)
            > _SAMPLING_RATE_TOLERANCE * bundle.sampling_rate_hz
        ):
            raise PreflightError(
                f"{caller}: {tag} samples at {actual_rate:g} Hz but "
                f"the concat preset is tuned for {bundle.sampling_rate_hz} "
                "Hz (the rate-keyed sorter row "
                f"{bundle.sorter_params_name!r} holds its clip_size / "
                "detect_interval snippet window at that rate). Every member "
                "must share the preset's acquisition rate."
            )


def _assert_concat_motion_stage(
    bundle,
    members: list[dict],
    group_key: dict,
    *,
    motion_mode,
    motion_correction_params_name: str,
    motion_estimate_id,
    manual_excluded_times,
    caller: str,
) -> None:
    """Raise ``PreflightError`` if a concat run's motion stage cannot run.

    The recipe must resolve, the preset's preprocessing recipe must filter,
    every member's geometry must support the estimation recipe, an
    ``"apply"`` sort must not run the sorter's own motion correction, and a
    supplied ``motion_estimate_id`` must match the concatenation the run
    would build.
    """
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.motion import (
        preprocessing_filter_problem,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters

    try:
        motion_recipe = resolve_motion_recipe(motion_correction_params_name)
    except ValueError as exc:
        raise PreflightError(f"{caller}: {exc}") from exc
    # The concatenation is built with the preset's preprocessing recipe
    # (checked to exist by assert_preset_compute_rows, which
    # assert_concat_preflight runs before this function).
    problem = preprocessing_filter_problem(bundle.preprocessing_params_name)
    if problem is not None:
        raise PreflightError(
            f"{caller}: motion_mode={motion_mode!r}: {problem}"
        )
    for member in members:
        problem = motion_geometry_problem(
            member["nwb_file_name"],
            int(member["sort_group_id"]),
            motion_recipe.resolved_estimation,
        )
        if problem is not None:
            raise PreflightError(
                f"{caller}: concat member {member['member_index']} "
                f"{problem}"
            )
    if motion_mode == "apply":
        sorter_params = (
            SorterParameters
            & {
                "sorter": bundle.sorter,
                "sorter_params_name": bundle.sorter_params_name,
            }
        ).fetch1("params")
        problem = sorter_motion_correction_problem(
            bundle.sorter, sorter_params, bundle.sorter_params_name
        )
        if problem is not None:
            raise PreflightError(f"{caller}: {problem}")
    if motion_estimate_id is not None:
        _assert_supplied_concat_estimate(
            motion_estimate_id,
            motion_recipe,
            bundle,
            members,
            group_key,
            manual_excluded_times=manual_excluded_times,
            caller=caller,
        )


def _assert_supplied_concat_estimate(
    motion_estimate_id,
    motion_recipe: "MotionRecipe",
    bundle,
    members: list[dict],
    group_key: dict,
    *,
    manual_excluded_times,
    caller: str,
) -> None:
    """Raise ``PreflightError`` unless the estimate matches this concat run.

    It must be a populated estimate, made with the recipe's estimation row,
    of a concatenation of this session group under the preset's
    preprocessing recipe whose frozen members are the member recordings and
    artifact detections this run would select.
    """
    from spyglass.spikesorting.v2.exceptions import PreflightError

    # The member recordings and masks the run would select, derived
    # like the single-session preview (manual exclusions per member).
    concat_members = []
    for member in members:
        recording_id, artifact_detection_id = expected_source_ids(
            bundle,
            nwb_file_name=member["nwb_file_name"],
            sort_group_id=int(member["sort_group_id"]),
            interval_list_name=member["interval_list_name"],
            team_name=member["team_name"],
            manual_excluded_times=(manual_excluded_times or {}).get(
                int(member["member_index"]), []
            ),
        )
        concat_members.append(
            {
                "member_index": int(member["member_index"]),
                "recording_id": recording_id,
                "artifact_detection_id": artifact_detection_id,
            }
        )
    problem = supplied_motion_estimate_problem(
        motion_estimate_id,
        motion_recipe,
        concat_source={
            **group_key,
            "preprocessing_params_name": (bundle.preprocessing_params_name),
        },
        concat_members=concat_members,
    )
    if problem is not None:
        raise PreflightError(f"{caller}: {problem}")


# A preflight message is read in a terminal, so it names enough contacts to
# act on and then says how many it left out. A 128-channel group with no
# geometry would otherwise print 128 triples.
_MAX_REPORTED_CONTACTS = 4


def _truncated(items: list) -> str:
    """Render ``items``, capped, stating how many were omitted."""
    shown = items[:_MAX_REPORTED_CONTACTS]
    omitted = len(items) - len(shown)
    return f"{shown}" + (f" (+{omitted} more)" if omitted else "")


def _missing_coordinate_report(channel_ids, positions) -> str:
    """Name the electrodes with a non-finite coordinate, and which one.

    "electrode 12 is missing rel_z" is actionable; "electrode 12 has no
    position" sends the operator looking for a row that is there. A whole
    unpositioned contact reports all three columns.

    Parameters
    ----------
    channel_ids : sequence of int
        Sort-group electrode ids, row-aligned to ``positions``.
    positions : numpy.ndarray
        ``(n, 3)`` contact positions in ``(rel_x, rel_y, rel_z)`` order, at
        least one coordinate of which is non-finite.

    Returns
    -------
    str
        ``"electrode 12 (rel_z); electrode 15 (rel_y, rel_z)"``, capped by
        ``_MAX_REPORTED_CONTACTS``.
    """
    import numpy as np

    columns = ("rel_x", "rel_y", "rel_z")
    finite = np.isfinite(np.asarray(positions, dtype=float))
    described = []
    for row in np.flatnonzero(~finite.all(axis=1)):
        absent = ", ".join(
            column
            for column, is_finite in zip(columns, finite[row])
            if not is_finite
        )
        described.append(f"electrode {int(channel_ids[row])} ({absent})")
    shown = described[:_MAX_REPORTED_CONTACTS]
    omitted = len(described) - len(shown)
    return "; ".join(shown) + (f" (+{omitted} more)" if omitted else "")


def _coincident_contact_report(channel_ids, positions) -> str:
    """Describe the electrodes that share an x-y position.

    Only called once ``select_distinct_plane`` has returned ``None``, so x-y
    necessarily has at least one duplicate: had it not, ``"xy"`` -- the first
    candidate plane -- would have been chosen.

    Parameters
    ----------
    channel_ids : sequence of int
        Sort-group electrode ids, row-aligned to ``positions``.
    positions : numpy.ndarray
        ``(n, 3)`` finite contact positions.

    Returns
    -------
    str
        ``"electrodes [1, 2] at (0.0, 0.0); ..."``, capped by
        ``_MAX_REPORTED_CONTACTS`` groups.
    """
    import numpy as np

    # Same rounding the geometry helpers use to decide "same contact", so the
    # message cannot name a different set than the verdict was based on.
    from spyglass.spikesorting.v2._recording_geometry import (
        _POSITION_DECIMALS,
    )

    xy = np.round(np.asarray(positions, dtype=float)[:, :2], _POSITION_DECIMALS)
    groups: dict = {}
    for electrode_id, position in zip(channel_ids, xy):
        groups.setdefault(tuple(position.tolist()), []).append(
            int(electrode_id)
        )
    colliding = [
        (position, ids) for position, ids in groups.items() if len(ids) > 1
    ]
    shown = colliding[:_MAX_REPORTED_CONTACTS]
    report = "; ".join(
        f"electrodes {_truncated(ids)} at {position}" for position, ids in shown
    )
    omitted = len(colliding) - len(shown)
    return report + (f" (+{omitted} more position(s))" if omitted else "")


def sort_group_geometry_problem(
    nwb_file_name: str, sort_group_id: int
) -> "str | None":
    """Report a sort group whose effective 2D contact geometry collapses.

    SpikeInterface builds a probe from the contact positions and rejects two
    contacts at the same place, so a sort group whose electrodes share a
    position fails minutes into ``Recording.populate`` (or, on a stored
    artifact, at the analyzer build). This reproduces the *effective* geometry
    the recording stage computes -- ``select_distinct_plane`` over the
    ``Probe.Electrode`` ``rel_x``/``rel_y``/``rel_z``, then the legacy
    ``tetrode_12.5`` repair -- and reports the failure up front instead.

    How much of the geometry is missing is classified by
    :func:`~spyglass.spikesorting.v2._recording_geometry.classify_missing_geometry`
    -- the same predicate ``normalize_channel_locations`` uses at make, so
    this check cannot clear a group that then fails there. NULL / absent
    ``rel_*`` for EVERY coordinate of the whole group is the legacy "geometry
    was never written" case: the raw electrodes table reads back as all-zero,
    which is exactly what the tetrode repair covers, so it is checked as
    all-zero rather than rejected outright. Anything less than that -- one
    NULL ``rel_z`` across the group, or a single unpositioned electrode among
    positioned ones -- is a partially-populated probe and is reported as such,
    naming the electrodes AND the missing columns (``select_distinct_plane``
    would otherwise raise on the NaN rows, and NaNs compare as distinct).

    Checked on the sort group's full electrode membership -- the same list
    ``Recording.make_fetch`` passes to ``maybe_apply_tetrode_geometry``. A
    ``bad_channel_handling='remove'`` run drops members later, which can only
    remove a collision, so this is conservative by at most that case.

    Reads the REGISTERED PROBE's geometry (``Probe.Electrode``, keyed by probe
    type), while the sort itself gets its channel locations from the session's
    own electrodes table. Those agree for any file whose writer filled both
    from one probe definition, but nothing enforces it -- see
    :func:`~spyglass.spikesorting.v2._recording_geometry.fetch_sort_group_contact_positions`.
    A session that disagrees with its probe can therefore pass this check and
    still reach the recording stage's own assertion, which sees what
    SpikeInterface sees.

    Parameters
    ----------
    nwb_file_name : str
        The raw NWB file.
    sort_group_id : int
        The sort group to check.

    Returns
    -------
    str or None
        ``None`` when the group's contacts are distinct (or will be made so
        by the tetrode repair); otherwise the operator-facing fix text.
    """
    from spyglass.spikesorting.v2._recording_geometry import (
        effective_contact_plane,
        fetch_sort_group_contact_positions,
        fetch_sort_group_probe_info,
        tetrode_repair_applies,
    )
    from spyglass.spikesorting.v2.recording import SortGroupV2

    channel_ids = sorted(
        (
            SortGroupV2.SortGroupElectrode
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": int(sort_group_id),
            }
        ).fetch("electrode_id"),
        key=int,
    )
    if not channel_ids:
        # ``sort_group_has_electrodes`` owns this failure; reporting it twice
        # would just duplicate the error.
        return None

    registered = fetch_sort_group_contact_positions(nwb_file_name, channel_ids)
    # The same verdict ``normalize_channel_locations`` reaches at make, so a
    # group cannot clear preflight and then fail there.
    verdict, positions, _ = effective_contact_plane(registered)
    if verdict == "partial":
        report = _missing_coordinate_report(channel_ids, positions)
        return (
            f"sort_group_id={int(sort_group_id)} of {nwb_file_name!r} has "
            "electrode(s) with an incomplete Probe.Electrode position "
            f"(missing row or NULL rel_*): {report}. SpikeInterface cannot "
            "place those contacts, and the tetrode_12.5 repair applies only "
            "to a group with NO positions at all. Populate Probe.Electrode "
            "rel_x/rel_y/rel_z for every electrode in the sort group."
        )

    if verdict == "plane":
        return None

    probe_types, electrode_group_names = fetch_sort_group_probe_info(
        nwb_file_name, channel_ids
    )
    if tetrode_repair_applies(
        probe_types, electrode_group_names, len(channel_ids)
    ):
        # The recording stage spreads these onto the 12.5 um square.
        return None
    return (
        f"sort_group_id={int(sort_group_id)} of {nwb_file_name!r} has "
        "contacts that share a position: no coordinate plane separates them. "
        f"Coincident in x-y (the projection SpikeInterface builds its probe "
        f"from): {_coincident_contact_report(channel_ids, positions)}. Fix "
        "Probe.Electrode rel_x/rel_y/rel_z for this sort group's electrodes; "
        "the tetrode_12.5 repair covers only 4-channel single-group tetrodes."
    )


class MotionRecipe(NamedTuple):
    """A ``MotionCorrectionParameters`` recipe with its two stage recipes.

    Attributes
    ----------
    recipe : dict
        The ``MotionCorrectionParameters`` row.
    estimation_params : dict
        The named ``MotionEstimationParameters`` row's ``params`` blob.
    interpolation_params : dict
        The named ``MotionInterpolationParameters`` row's ``params`` blob.
    resolved_estimation : dict
        ``estimation_params`` resolved against the installed SpikeInterface
        (``_motion.resolve_estimation_params``).
    """

    recipe: dict
    estimation_params: dict
    interpolation_params: dict
    resolved_estimation: dict


def resolve_motion_recipe(motion_correction_params_name: str) -> MotionRecipe:
    """Fetch a motion-correction recipe and resolve its estimation recipe.

    Parameters
    ----------
    motion_correction_params_name : str
        The ``MotionCorrectionParameters`` row.

    Returns
    -------
    MotionRecipe

    Raises
    ------
    ValueError
        If the row is missing, or its estimation recipe no longer resolves
        against the installed SpikeInterface.
    """
    from spyglass.spikesorting.v2._motion import resolve_estimation_params
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectionParameters,
        MotionEstimationParameters,
        MotionInterpolationParameters,
    )

    rows = (
        MotionCorrectionParameters
        & {"motion_correction_params_name": motion_correction_params_name}
    ).fetch(as_dict=True)
    if len(rows) != 1:
        raise ValueError(
            "MotionCorrectionParameters row "
            f"{motion_correction_params_name!r} is missing. Run "
            "initialize_v2_defaults() (it ships 'dredge_v1' and "
            "'dredge_fast_v1'), or insert the recipe first."
        )
    recipe = rows[0]
    estimation_name = recipe["motion_estimation_params_name"]
    estimation_params = (
        MotionEstimationParameters
        & {"motion_estimation_params_name": estimation_name}
    ).fetch1("params")
    interpolation_params = (
        MotionInterpolationParameters
        & {
            "motion_interpolation_params_name": recipe[
                "motion_interpolation_params_name"
            ]
        }
    ).fetch1("params")
    try:
        resolved = resolve_estimation_params(estimation_params)
    except ValueError as exc:
        raise ValueError(
            f"MotionEstimationParameters row {estimation_name!r} (named by "
            f"motion recipe {motion_correction_params_name!r}) does not "
            f"resolve against the installed SpikeInterface: {exc}"
        ) from exc
    return MotionRecipe(
        recipe, estimation_params, interpolation_params, resolved
    )


def sorter_motion_correction_problem(
    sorter: str, params, sorter_params_name: str
) -> "str | None":
    """Say why a sorter row cannot sort a motion-corrected recording.

    Delegates to ``_params.sorter.reject_internal_motion_correction`` (the
    check ``SortingSelection.insert_selection`` runs), so preflight and the
    insert agree.

    Parameters
    ----------
    sorter, params, sorter_params_name
        The ``SorterParameters`` row.

    Returns
    -------
    str or None
        The problem, naming the row and the key to turn off, or ``None``.
    """
    from spyglass.spikesorting.v2._params.sorter import (
        reject_internal_motion_correction,
    )

    try:
        reject_internal_motion_correction(
            sorter, params, sorter_params_name=sorter_params_name
        )
    except ValueError as exc:
        return f"motion_mode='apply' sorts a motion-corrected recording: {exc}"
    return None


def _concat_member_mismatch(concat_key: dict, expected: list[dict]) -> list:
    """Describe how a concatenation's frozen members differ from a run's.

    Parameters
    ----------
    concat_key : dict
        ``{"concat_recording_id": ...}`` of the saved concatenation.
    expected : list of dict
        The run's members: ``member_index``, ``recording_id`` and
        ``artifact_detection_id`` (``None`` for an unmasked member).

    Returns
    -------
    list of str
        One description per differing member set or field; empty when they
        agree.
    """
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    def _id(value):
        return None if value is None else uuid.UUID(str(value))

    fields = ("recording_id", "artifact_detection_id")
    frozen = {
        int(row["member_index"]): row
        for row in (
            ConcatenatedRecordingSelection.MemberSnapshot & concat_key
        ).fetch("member_index", *fields, as_dict=True)
    }
    wanted = {int(row["member_index"]): row for row in expected}
    if set(frozen) != set(wanted):
        return [
            f"its concatenation has members {sorted(frozen)}, but this "
            f"run's has {sorted(wanted)}"
        ]
    return [
        f"member {index}: its {name} {_id(frozen[index][name])} != the run's "
        f"{_id(wanted[index][name])}"
        for index in sorted(wanted)
        for name in fields
        if _id(frozen[index][name]) != _id(wanted[index][name])
    ]


def expected_source_ids(
    bundle,
    *,
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    manual_excluded_times,
) -> tuple:
    """Preview the recording and artifact-detection ids a run would mint.

    Uses the same DB-free builders ``RecordingSelection.insert_selection`` and
    ``RecordingArtifactSelection.insert_selection`` use, so the preview cannot
    drift from the insert; only the recording's input hash is resolved from
    the live sort-group, electrode and preprocessing rows (they must exist).

    Parameters
    ----------
    bundle : _PipelinePreset
        The preset (its preprocessing and artifact rows).
    nwb_file_name, sort_group_id, interval_list_name, team_name
        The recording's selection fields.
    manual_excluded_times : list
        The normalized manual exclusions of this recording.

    Returns
    -------
    recording_id : uuid.UUID
    artifact_detection_id : uuid.UUID or None
        ``None`` when the preset runs no artifact detection.
    """
    from spyglass.spikesorting.v2._selection_identity import (
        artifact_detection_identity_payload,
        deterministic_id,
    )
    from spyglass.spikesorting.v2._selection_plan import (
        build_recording_selection_plan,
    )
    from spyglass.spikesorting.v2.recording import resolve_recording_input_hash

    recording_id = build_recording_selection_plan(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": interval_list_name,
            "preprocessing_params_name": bundle.preprocessing_params_name,
            "team_name": team_name,
        },
        recording_input_hash=resolve_recording_input_hash(
            nwb_file_name, sort_group_id, bundle.preprocessing_params_name
        ),
    ).recording_id
    # A None artifact name means no artifact-detection pass: the sort's
    # identity carries artifact_detection_id=None (matching
    # build_sorting_selection_plan), so there is no RecordingArtifactSelection
    # PK to expect.
    if bundle.artifact_detection_params_name is None:
        return recording_id, None
    return recording_id, deterministic_id(
        "artifact_detection",
        artifact_detection_identity_payload(
            artifact_detection_params_name=bundle.artifact_detection_params_name,
            recording_id=recording_id,
            manual_excluded_times=manual_excluded_times,
        ),
    )


def supplied_motion_estimate_problem(
    motion_estimate_id,
    motion_recipe: MotionRecipe,
    *,
    source_lineage=None,
    concat_source: "dict | None" = None,
    concat_members: "list[dict] | None" = None,
) -> "str | None":
    """Say why a saved motion estimate cannot be applied by a run.

    The estimate must be populated, estimated with the recipe's estimation
    row, on a source whose traces have not changed since (its live
    ``content_hash``, when the source row exists, is the one the estimate
    was selected on), and made from the run's source: exactly its source and artifact
    mask (``source_lineage``), or, before a concat run has built its
    concatenation, a concatenation of the same session group under the same
    preprocessing recipe (``concat_source``) whose frozen members are the
    run's members, recordings and artifact detections (``concat_members``).

    Parameters
    ----------
    motion_estimate_id : uuid.UUID or str
        The ``MotionEstimate`` to apply.
    motion_recipe : MotionRecipe
        The run's resolved ``MotionCorrectionParameters`` recipe.
    source_lineage : SourceLineage, optional
        The run's source and artifact detection.
    concat_source : dict, optional
        ``session_group_owner``, ``session_group_name`` and
        ``preprocessing_params_name`` of a concat run.
    concat_members : list of dict, optional
        The concat run's expected members, each ``member_index``,
        ``recording_id`` and ``artifact_detection_id`` (``None`` unmasked),
        compared with the estimate's concatenation's ``MemberSnapshot``.

    Returns
    -------
    str or None
        Every mismatch, each naming the estimate's value and the run's, or
        ``None`` when the estimate can be applied.
    """
    from spyglass.spikesorting.v2._source_resolution import (
        correction_lineage_mismatch,
    )
    from spyglass.spikesorting.v2.motion import (
        _SOURCE_TABLES,
        MotionEstimate,
        MotionEstimateSelection,
        _live_source_row,
    )

    key = {"motion_estimate_id": uuid.UUID(str(motion_estimate_id))}
    if not (MotionEstimate & key):
        return (
            f"motion_estimate_id {key['motion_estimate_id']} is not a "
            "populated MotionEstimate. Save the estimate first "
            "(MotionEstimateSelection.insert_selection, then "
            "MotionEstimate.populate)."
        )
    problems = []
    estimation_name = (MotionEstimateSelection & key).fetch1(
        "motion_estimation_params_name"
    )
    recipe_estimation = motion_recipe.recipe["motion_estimation_params_name"]
    if estimation_name != recipe_estimation:
        problems.append(
            f"it was estimated with MotionEstimationParameters "
            f"{estimation_name!r}, but recipe "
            f"{motion_recipe.recipe['motion_correction_params_name']!r} "
            f"names {recipe_estimation!r}"
        )
    lineage = MotionEstimateSelection.resolve_source(key)
    # A source recomputed since the estimate was selected no longer has the
    # traces the estimate describes (checked while the source row exists; a
    # run that has not built it yet is checked again when it applies it).
    if _SOURCE_TABLES[lineage.kind] & lineage.key:
        try:
            _live_source_row(
                lineage,
                key["motion_estimate_id"],
                (MotionEstimateSelection & key).fetch1("source_content_hash"),
            )
        except ValueError as exc:
            problems.append(str(exc))
    if source_lineage is not None:
        problems.extend(
            correction_lineage_mismatch(source_lineage, lineage, consumer="run")
        )
    if concat_source is not None:
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecordingSelection,
        )

        if lineage.kind != "concatenated_recording":
            problems.append(
                f"it was estimated on recording {lineage.key}, but this run "
                "sorts a concatenation"
            )
        else:
            estimated = (ConcatenatedRecordingSelection & lineage.key).fetch1(
                *concat_source
            )
            for name, value in zip(concat_source, estimated):
                if value != concat_source[name]:
                    problems.append(
                        f"its concatenation has {name} {value!r}, but this "
                        f"run's has {concat_source[name]!r}"
                    )
            if concat_members is not None:
                problems.extend(
                    _concat_member_mismatch(lineage.key, concat_members)
                )
    if not problems:
        return None
    return (
        f"motion estimate {key['motion_estimate_id']} cannot be applied to "
        f"this run: {'; '.join(problems)}. Apply an estimate of this run's "
        "source, artifact mask and recipe."
    )


def motion_geometry_problem_from_contacts(
    channel_ids,
    positions,
    probe_types,
    electrode_group_names,
    probe_shanks,
    resolved_estimation: dict,
) -> "str | None":
    """Say why a sort group's effective geometry cannot be motion-estimated.

    Rebuilds the planar geometry the recording stage produces from the
    registered ``Probe.Electrode`` positions -- ``select_distinct_plane``, or
    the legacy ``tetrode_12.5`` repair for a group with no usable positions --
    on a one-frame stand-in recording, and runs the estimator's own
    eligibility check (``_motion.check_estimation_eligibility``: finite,
    distinct positions, one shank, an attachable probe, a depth extent of at
    least the detection radius, and room for the nonrigid windows). A group
    whose geometry is incomplete or collapses without the repair is left to
    ``sort_group_geometry_problem`` (``None`` here), so it is not reported
    twice.

    Parameters
    ----------
    channel_ids : sequence of int
        Sort-group electrode ids, sorted, row-aligned to the other inputs.
    positions : array_like
        ``(n, 3)`` ``(rel_x, rel_y, rel_z)`` contact positions (NaN where
        missing).
    probe_types, electrode_group_names, probe_shanks : sequence
        Per-channel probe type, electrode group and probe shank.
    resolved_estimation : dict
        A resolved estimation recipe (``_motion.resolve_estimation_params``).

    Returns
    -------
    str or None
        The estimator's refusal, or ``None`` when the geometry is eligible.
    """
    import numpy as np
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._motion import check_estimation_eligibility
    from spyglass.spikesorting.v2._recording_geometry import (
        effective_contact_plane,
        maybe_apply_tetrode_geometry,
        tetrode_repair_applies,
    )

    channel_ids = [int(c) for c in channel_ids]
    verdict, _, planar = effective_contact_plane(positions)
    repaired = verdict == "collapsed" and tetrode_repair_applies(
        tuple(probe_types), tuple(electrode_group_names), len(channel_ids)
    )
    if verdict != "plane" and not repaired:
        return None
    recording = NumpyRecording(
        np.zeros((1, len(channel_ids)), dtype="float32"),
        sampling_frequency=30000.0,
        channel_ids=channel_ids,
    )
    if repaired:
        maybe_apply_tetrode_geometry(
            recording,
            tuple(probe_types),
            tuple(electrode_group_names),
            channel_ids,
        )
    else:
        recording.set_channel_locations(planar)
    recording.set_property("group", [str(g) for g in electrode_group_names])
    recording.set_property("probe_shank", [str(s) for s in probe_shanks])
    try:
        check_estimation_eligibility(recording, resolved_estimation)
    except ValueError as exc:
        return str(exc)
    return None


def motion_geometry_problem(
    nwb_file_name: str, sort_group_id: int, resolved_estimation: dict
) -> "str | None":
    """Say why a sort group cannot be motion-estimated with a recipe.

    Reads the group's electrodes, registered contact positions, probe types,
    electrode groups and shanks, then applies
    :func:`motion_geometry_problem_from_contacts`. Evaluated on the full
    electrode membership, like :func:`sort_group_geometry_problem`; the
    estimator re-checks the actual recording.

    Parameters
    ----------
    nwb_file_name : str
    sort_group_id : int
    resolved_estimation : dict
        A resolved estimation recipe.

    Returns
    -------
    str or None
        The operator-facing problem, or ``None``.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2._recording_geometry import (
        fetch_sort_group_contact_positions,
        fetch_sort_group_probe_info,
    )
    from spyglass.spikesorting.v2.recording import SortGroupV2

    channel_ids = sorted(
        int(c)
        for c in (
            SortGroupV2.SortGroupElectrode
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": int(sort_group_id),
            }
        ).fetch("electrode_id")
    )
    if not channel_ids:
        return None
    positions = fetch_sort_group_contact_positions(nwb_file_name, channel_ids)
    probe_types, group_names = fetch_sort_group_probe_info(
        nwb_file_name, channel_ids
    )
    shanks = (
        Electrode
        & {"nwb_file_name": nwb_file_name}
        & [{"electrode_id": c} for c in channel_ids]
    ).fetch("probe_shank", order_by="electrode_id")
    problem = motion_geometry_problem_from_contacts(
        channel_ids,
        positions,
        probe_types,
        group_names,
        shanks,
        resolved_estimation,
    )
    if problem is None:
        return None
    return (
        f"sort_group_id={int(sort_group_id)} of {nwb_file_name!r} cannot be "
        f"motion-estimated with this recipe: {problem}"
    )


def preflight_v2_pipeline(
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    pipeline_preset: str = DEFAULT_PIPELINE_PRESET,
    auto_curate: bool = False,
    manual_excluded_times=None,
    motion_mode: MotionMode = "off",
    motion_correction_params_name: "str | None" = None,
    motion_estimate_id=None,
    *,
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
    from spyglass.spikesorting.v2._manual_artifacts import (
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
    sorter_params_query, sorter_params_exist, sorter_row = (
        _check_sorter_param_rows(_check, bundle, sort_checks=sort_checks)
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
            sorter_params_query,
            sorter_params_exist,
        )
    motion_recipe = None
    if motion_mode != "off":
        motion_recipe = _check_motion_stage(
            _check,
            bundle,
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            motion_mode=motion_mode,
            motion_correction_params_name=motion_correction_params_name,
            preprocessing_params_exist=preprocessing_params_exist,
            sort_group_exists=sort_group_exists,
            sorter_params_exist=sorter_params_exist,
            sorter_row=sorter_row,
        )

    # Non-blocking advisory: the "none" artifact params are a no-op
    # pass-through (no masking). "default" performs real amplitude-threshold
    # detection and is the legitimate built-in choice, so it is NOT warned.
    if bundle.artifact_detection_params_name in (None, "none"):
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

    errors = [c.fix for c in checks if not c.ok]
    resource_notes = _group_resource_notes(
        display_waveform_params_name,
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        effective_config=effective_config,
    )
    return PreflightReport(
        ok=not errors,
        errors=errors,
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
        ok=False,
        errors=[c.fix for c in checks if not c.ok],
        warnings=warnings,
        resolved_pipeline_preset=pipeline_preset,
        expected_ids={},
        checks=checks,
    )


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


def _missing_row_fix(row_description: str) -> str:
    """The fix for a missing shipped parameter row named by ``row_description``."""
    return f"{row_description} is missing. Run initialize_v2_defaults()."


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
        _missing_row_fix(
            f"PreprocessingParameters row {bundle.preprocessing_params_name!r}"
        ),
    )
    # An explicit no-mask preset has no artifact parameter row to require.
    if bundle.artifact_detection_params_name is not None:
        check(
            "artifact_detection_params_exist",
            ArtifactDetectionParameters
            & {
                "artifact_detection_params_name": bundle.artifact_detection_params_name
            },
            _missing_row_fix(
                "ArtifactDetectionParameters row "
                f"{bundle.artifact_detection_params_name!r}"
            ),
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
    sorter_params_query : QueryExpression
        The ``SorterParameters`` restriction to the preset's row.
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
        _missing_row_fix(
            f"SorterParameters row (sorter={bundle.sorter!r}, "
            f"sorter_params_name={bundle.sorter_params_name!r})"
        ),
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
    return sorter_params_query, sorter_params_exist, sorter_row


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
    from spyglass.spikesorting.v2._recipe_catalog import (
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
            _missing_row_fix(
                "AnalyzerWaveformParameters row "
                f"{display_waveform_params_name!r} (the display analyzer "
                f"recipe for preprocessing {bundle.preprocessing_params_name!r})"
            ),
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
    from spyglass.spikesorting.v2._recipe_catalog import (
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
        _missing_row_fix(
            f"QualityMetricParameters row {bundle.metric_params_name!r} (the "
            "auto-curation metric set)"
        ),
    )
    check(
        "auto_curation_rules_exist",
        AutoCurationRules
        & {"auto_curation_rules_name": bundle.auto_curation_rules_name},
        _missing_row_fix(
            f"AutoCurationRules row {bundle.auto_curation_rules_name!r} (the "
            "auto-curation rule set)"
        ),
    )
    metric_waveform_params_name = waveform_params_for_preprocessing(
        bundle.preprocessing_params_name
    )[1]
    check(
        "metric_waveform_params_exist",
        AnalyzerWaveformParameters
        & {"waveform_params_name": metric_waveform_params_name},
        _missing_row_fix(
            f"AnalyzerWaveformParameters row {metric_waveform_params_name!r} "
            "(the whitened metric analyzer recipe auto-curation scores on)"
        ),
    )


def _check_sampling_rate(
    check: "Callable[[str, Any, str], bool]",
    bundle,
    nwb_file_name: str,
    pipeline_preset: str,
) -> None:
    """Check the recording samples at the rate the preset is tuned for.

    The MS4/MS5 snippet window (clip_size / detect_interval on the rate-keyed
    sorter row) assumes a specific acquisition rate, so a 30 kHz preset on a
    20 kHz recording (or the reverse) silently sorts with a mistuned window.
    The clusterless preset is rate-agnostic (sampling_rate_hz is None) and is
    skipped; the check is also skipped if Raw is not ingested yet (raw_exists
    already reports that, so this would only add a confusing second failure).
    """
    from spyglass.common import Raw

    if bundle.sampling_rate_hz is not None:
        raw = Raw & {"nwb_file_name": nwb_file_name}
        if raw:
            actual_rate = float(raw.fetch1("sampling_rate"))
            rate_ok = (
                abs(actual_rate - bundle.sampling_rate_hz)
                <= _SAMPLING_RATE_TOLERANCE * bundle.sampling_rate_hz
            )
            check(
                "sampling_rate_matches",
                rate_ok,
                f"recording {nwb_file_name!r} samples at {actual_rate:g} Hz "
                f"but pipeline_preset {pipeline_preset!r} is tuned for "
                f"{bundle.sampling_rate_hz} Hz: the rate-keyed sorter row "
                f"{bundle.sorter_params_name!r} holds its clip_size / "
                "detect_interval snippet window at that rate. Pick the "
                "rate-matched preset (call describe_pipeline_presets() and "
                "match sampling_rate_hz to the recording).",
            )


def _check_sorter_execution(
    check: "Callable[[str, Any, str], bool]",
    warnings: list[str],
    bundle,
    pipeline_preset: str,
    sorter_params_query,
    sorter_params_exist: bool,
) -> None:
    """Check the sorter can execute on the row's execution backend.

    The selected backend (local vs container) is read ONLY from the
    SorterParameters row's execution_params (the single source of truth --
    never the preset). When the row is absent, sorter_params_exist already
    reported the blocking error and the backend is unknowable, so the
    container / MATLAB-policy checks are skipped; the sorter NAME is still
    validated as a local sorter so a misspelled sorter keeps its spelling
    hint. A runnable container backend appends an advisory to ``warnings``.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import SorterParameters

    if not sorter_params_exist:
        _check_local_sorter_runtime(
            bundle, sis, SorterParameters._NON_SI_SORTERS, check
        )
    else:
        from spyglass.spikesorting.v2._params.sorter import (
            validate_execution_params,
        )
        from spyglass.spikesorting.v2._sorting_dispatch import (
            MATLAB_SORTERS,
            matlab_container_required_message,
        )

        execution_params = validate_execution_params(
            sorter_params_query.fetch1("execution_params")
        )
        execution_backend = execution_params["backend"]
        container_image = execution_params["container_image"]

        if execution_backend == "local":
            # MATLAB-backed sorters (Kilosort 2.5/3, IronClust) ship only as
            # container images; a local row for one of them cannot run. Surface
            # the SAME tracked-container-backend message the dispatch raises,
            # rather than the local-install checks.
            if bundle.sorter.lower() in MATLAB_SORTERS:
                check(
                    "sorter_execution_backend",
                    False,
                    matlab_container_required_message(bundle.sorter),
                )
            else:
                _check_local_sorter_runtime(
                    bundle, sis, SorterParameters._NON_SI_SORTERS, check
                )
        else:
            # Container backend: verify the container RUNTIME (engine + Python
            # package), not the local sorter install. The sorter runtime lives
            # in the image, so a missing LOCAL runtime is irrelevant here. A
            # missing CONTAINER runtime is an actionable, blocking
            # selected-preset error -- preflight never silently falls back to
            # local execution.
            runtime_ok, runtime_detail = _container_runtime_available(
                execution_backend
            )
            check(
                "container_runtime_available",
                runtime_ok,
                f"pipeline_preset {pipeline_preset!r} selects the "
                f"{execution_backend} execution backend (image "
                f"{container_image!r}) for sorter {bundle.sorter!r}, but it is "
                f"not runnable here: {runtime_detail}. Install the container "
                "runtime and its Python package, or pick a local-execution "
                "preset. Preflight does not fall back to local execution.",
            )
            # Informational advisory (only when the container is actually
            # runnable, so it never sits next to a blocking runtime failure):
            # the host can stay on numpy>=2 -- the sorter runtime (e.g. MS4's
            # numpy<2-era ml_ms4alg) lives in the container, not on the host.
            if runtime_ok:
                warnings.append(
                    f"pipeline_preset {pipeline_preset!r} runs sorter "
                    f"{bundle.sorter!r} inside the {execution_backend} image "
                    f"{container_image!r}: the host can stay on the v2 numpy>=2 "
                    "environment because the sorter runtime lives in the "
                    "container, not on the host."
                )


def _check_motion_stage(
    check: "Callable[[str, Any, str], bool]",
    bundle,
    *,
    nwb_file_name: str,
    sort_group_id: int,
    motion_mode,
    motion_correction_params_name: str,
    preprocessing_params_exist: bool,
    sort_group_exists: bool,
    sorter_params_exist: bool,
    sorter_row: "dict | None",
) -> "MotionRecipe | None":
    """Check a motion stage (estimate / apply) can run; return its recipe.

    The recipe must exist and resolve, the preset's preprocessing recipe must
    filter (motion is estimated on filtered, unwhitened traces;
    MotionEstimateSelection.insert_selection refuses the same source), the
    group's effective geometry must support the estimation recipe (checked
    here on the registered positions, before any populate; the estimator
    re-checks the actual recording), and a sort of a corrected recording must
    not run the sorter's own motion correction.

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
        if sort_group_exists:
            motion_geometry = motion_geometry_problem(
                nwb_file_name,
                sort_group_id,
                motion_recipe.resolved_estimation,
            )
            check(
                "motion_geometry_supported",
                motion_geometry is None,
                motion_geometry or "",
            )
    if motion_mode == "apply" and sorter_params_exist:
        sorter_motion = sorter_motion_correction_problem(
            bundle.sorter, sorter_row["params"], bundle.sorter_params_name
        )
        check(
            "sorter_motion_correction_off",
            sorter_motion is None,
            sorter_motion or "",
        )
    return motion_recipe


def _preflight_expected_ids(
    check: "Callable[[str, Any, str], bool]",
    bundle,
    *,
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    manual_excluded_times,
    motion_mode,
    motion_recipe: "MotionRecipe | None",
    motion_estimate_id,
) -> dict:
    """The selection PKs a run would produce, each with its existence flags.

    Also records the ``motion_estimate_applicable`` check for a supplied
    ``motion_estimate_id``, which needs the expected recording and artifact
    detection ids.

    Returns
    -------
    dict
        The :attr:`PreflightReport.expected_ids` mapping.
    """
    from spyglass.spikesorting.v2._selection_plan import (
        build_sorting_selection_plan,
    )
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    recording_id, artifact_detection_id = expected_source_ids(
        bundle,
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name=interval_list_name,
        team_name=team_name,
        manual_excluded_times=manual_excluded_times,
    )
    if motion_estimate_id is not None and motion_recipe is not None:
        from spyglass.spikesorting.v2._source_resolution import (
            SourceLineage,
        )

        supplied = supplied_motion_estimate_problem(
            motion_estimate_id,
            motion_recipe,
            source_lineage=SourceLineage(
                kind="recording",
                key={"recording_id": recording_id},
                artifact_detection_id=artifact_detection_id,
            ),
        )
        check("motion_estimate_applicable", supplied is None, supplied)
    motion_ids = _expected_motion_ids(
        motion_mode,
        motion_recipe,
        recording_id=recording_id,
        artifact_detection_id=artifact_detection_id,
        motion_estimate_id=motion_estimate_id,
    )
    corrected = motion_ids.get("motion_corrected_recording_id", {})
    if corrected.get("pending"):
        sorting_entry = _pending_id_entry(corrected["pending"])
    else:
        sorting_id = build_sorting_selection_plan(
            {
                "recording_id": recording_id,
                "sorter": bundle.sorter,
                "sorter_params_name": bundle.sorter_params_name,
                "artifact_detection_id": artifact_detection_id,
                "motion_corrected_recording_id": corrected.get("id"),
            }
        ).sorting_id
        sorting_entry = {
            "id": sorting_id,
            "exists": bool(SortingSelection & {"sorting_id": sorting_id}),
            "computed_exists": bool(Sorting & {"sorting_id": sorting_id}),
        }
    # Per stage, ``exists`` is whether the SELECTION row exists (the run
    # would reuse this PK) and ``computed_exists`` whether the COMPUTED
    # output row exists (the populate already ran -- a reused, near-zero-cost
    # stage). Distinguishing them tells a caller what work the run would
    # actually do: a selection can exist with its output not yet populated.
    return {
        "recording_id": {
            "id": recording_id,
            "exists": bool(RecordingSelection & {"recording_id": recording_id}),
            "computed_exists": bool(Recording & {"recording_id": recording_id}),
        },
        "artifact_detection_id": (
            {"id": None, "exists": False, "computed_exists": False}
            if artifact_detection_id is None
            else {
                "id": artifact_detection_id,
                "exists": bool(
                    RecordingArtifactSelection
                    & {"artifact_detection_id": artifact_detection_id}
                ),
                "computed_exists": bool(
                    RecordingArtifactDetection
                    & {"artifact_detection_id": artifact_detection_id}
                ),
            }
        ),
        **motion_ids,
        "sorting_id": sorting_entry,
    }


def _group_resource_notes(
    display_waveform_params_name: str,
    *,
    nwb_file_name: str,
    sort_group_id: int,
    effective_config: "dict | None",
) -> list[str]:
    """:func:`_resource_notes` for one sort group, read from its rows."""
    from spyglass.common import Raw
    from spyglass.spikesorting.v2.recording import SortGroupV2

    return _resource_notes(
        display_waveform_params_name,
        n_channels=len(
            SortGroupV2.SortGroupElectrode
            & {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}
        ),
        sampling_rate_hz=(
            float(
                (Raw & {"nwb_file_name": nwb_file_name}).fetch1("sampling_rate")
            )
            if Raw & {"nwb_file_name": nwb_file_name}
            else None
        ),
        effective_config=effective_config,
    )


def _pending_id_entry(reason: str) -> dict:
    """An ``expected_ids`` entry whose id cannot be derived yet, and why."""
    return {
        "id": None,
        "exists": False,
        "computed_exists": False,
        "pending": reason,
    }


def _expected_motion_ids(
    motion_mode,
    motion_recipe: "MotionRecipe | None",
    *,
    recording_id,
    artifact_detection_id,
    motion_estimate_id=None,
) -> dict:
    """Preview the motion selection ids a single-recording run would mint.

    The estimate id folds in the source recording's ``content_hash``, so it
    (and the corrected id and an ``"apply"`` sort id built on it) is known
    only once the ``Recording`` is computed; until then the entries carry
    ``id=None`` and a ``pending`` reason instead of a guess. A supplied
    ``motion_estimate_id`` (the saved estimate an ``"apply"`` run reuses) is
    known up front. Uses the DB-free derivations the selection inserts use,
    so a preview cannot drift from the insert.

    Returns
    -------
    dict
        ``{}`` for ``"off"`` or an unresolved recipe; otherwise
        ``motion_estimate_id`` (and, for ``"apply"``,
        ``motion_corrected_recording_id``) entries shaped like the other
        ``expected_ids`` entries.
    """
    if motion_mode == "off" or motion_recipe is None:
        return {}
    from spyglass.spikesorting.v2._motion import (
        motion_corrected_selection_identity,
        motion_estimate_selection_identity,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionCorrectedRecordingSelection,
        MotionEstimate,
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.recording import Recording

    names = ["motion_estimate_id"] + (
        ["motion_corrected_recording_id"] if motion_mode == "apply" else []
    )
    if motion_estimate_id is not None:
        estimate_id = uuid.UUID(str(motion_estimate_id))
    else:
        content_hashes = (Recording & {"recording_id": recording_id}).fetch(
            "content_hash"
        )
        if len(content_hashes) == 0:
            pending = (
                "the recording is computed (the motion estimate id includes "
                "its content hash)"
            )
            return {name: _pending_id_entry(pending) for name in names}
        estimate_id = motion_estimate_selection_identity(
            source_kind="recording",
            source_id=recording_id,
            source_content_hash=content_hashes[0],
            artifact_detection_id=artifact_detection_id,
            motion_estimation_params_name=motion_recipe.recipe[
                "motion_estimation_params_name"
            ],
            estimation_params=motion_recipe.estimation_params,
        ).selection_id
    estimate_key = {"motion_estimate_id": estimate_id}
    expected = {
        "motion_estimate_id": {
            "id": estimate_id,
            "exists": bool(MotionEstimateSelection & estimate_key),
            "computed_exists": bool(MotionEstimate & estimate_key),
        }
    }
    if motion_mode == "apply":
        corrected_id = motion_corrected_selection_identity(
            motion_estimate_id=estimate_id,
            motion_interpolation_params_name=motion_recipe.recipe[
                "motion_interpolation_params_name"
            ],
            interpolation_params=motion_recipe.interpolation_params,
        ).selection_id
        corrected_key = {"motion_corrected_recording_id": corrected_id}
        expected["motion_corrected_recording_id"] = {
            "id": corrected_id,
            "exists": bool(MotionCorrectedRecordingSelection & corrected_key),
            "computed_exists": bool(MotionCorrectedRecording & corrected_key),
        }
    return expected


def _resource_notes(
    display_waveform_params_name: str,
    *,
    n_channels: int,
    sampling_rate_hz: "float | None",
    effective_config: "dict | None",
) -> list[str]:
    """Known allocations / scratch / worker settings, from tracked recipes.

    Pre-sort the unit count is unknown, so the waveform bound is stated per
    unit; multiply by the expected unit count for the buffer size. Channels
    per unit are bounded by the sort group's channel count (radius / best
    channels sparsity reduces it; dense uses all).
    """
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_cache_root
    from spyglass.spikesorting.v2._params.analyzer_waveform import (
        SparsityParams,
    )
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters
    from spyglass.settings import temp_dir

    notes: list[str] = []
    rows = (
        AnalyzerWaveformParameters
        & {"waveform_params_name": display_waveform_params_name}
    ).fetch(as_dict=True)
    if len(rows) == 1 and sampling_rate_hz:
        params = dict(rows[0]["params"])
        sparsity = SparsityParams.model_validate(params.get("sparsity") or {})
        n_samples = int(
            round(
                (float(params["ms_before"]) + float(params["ms_after"]))
                * sampling_rate_hz
                / 1000.0
            )
        )
        if sparsity.method == "best_channels":
            channels = min(int(sparsity.num_channels), int(n_channels))
            channel_note = f"{channels} channels (best_channels)"
        elif sparsity.method == "radius":
            channels = int(n_channels)
            channel_note = (
                f"<= {channels} channels (radius {sparsity.radius_um:g} um; "
                "fewer on wide groups)"
            )
        else:
            channels = int(n_channels)
            channel_note = f"{channels} channels (dense)"
        per_unit = int(params["max_spikes_per_unit"]) * n_samples * channels * 4
        notes.append(
            "display analyzer waveforms: up to "
            f"{int(params['max_spikes_per_unit'])} spikes/unit x {n_samples} "
            f"samples x {channel_note} x 4 bytes = up to "
            f"{per_unit / 1024**2:.1f} MiB per unit on disk (memmapped; "
            "extraction peak ~1.5x the total, loads are lazy)."
        )
    notes.append(
        f"analyzer cache root: {analyzer_cache_root()} (display analyzer, plus "
        "a whitened metric analyzer when PC metrics are requested, plus one "
        "per-generation cache per reviewed merged curation)."
    )
    notes.append(
        f"sorter scratch: a per-sort temporary directory under {temp_dir}, "
        "removed when the sort finishes; the preprocessed Recording is a "
        "separate NWB artifact."
    )
    if effective_config:
        jobs = effective_config.get("job_kwargs") or {}
        notes.append(
            "effective worker/chunk settings (sort + analyzer): "
            f"n_jobs={jobs.get('n_jobs', 1)}, "
            f"chunk_duration={jobs.get('chunk_duration', '1s')!r}, "
            f"random_seed={effective_config.get('random_seed')}, "
            f"external_whiten={effective_config.get('external_whiten')}."
        )
    return notes


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
