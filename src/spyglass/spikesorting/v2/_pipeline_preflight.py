"""Pre-populate configuration checks, extracted from ``pipeline.py``.

Behavior-preserving: the ``Preflight*`` dataclasses, the
``_SORTER_RUNTIME_BACKENDS`` map, ``preflight_v2_pipeline``,
``_resolve_session_sort_group_ids``, and ``preflight_v2_pipeline_session``
move here verbatim. ``pipeline.py`` re-exports the public names so user
import paths are unchanged. Depends only on ``_pipeline_presets``
(``_PIPELINE_PRESETS``); the run module depends on this one (run -> preflight).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from spyglass.spikesorting.v2._pipeline_presets import _PIPELINE_PRESETS
from spyglass.spikesorting.v2._recipe_catalog import DEFAULT_PIPELINE_PRESET

if TYPE_CHECKING:
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


def _check_local_sorter_runtime(bundle, sis, non_si_sorters, check) -> None:
    """Run the LOCAL-execution sorter checks (installed + runtime backend).

    Extracted from ``preflight_v2_pipeline`` so its execution-backend dispatch
    reads as a flat three-arm choice (MATLAB-local error / local checks /
    container runtime) instead of a deeply nested branch.

    Parameters
    ----------
    bundle : _PipelinePreset
        The resolved preset whose ``sorter`` is being checked.
    sis : module
        ``spikeinterface.sorters`` (passed in so the import stays at the one
        call site).
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
        ``CurationV2.insert_curation``, not content-addressed.
    checks
        Per-check detail; every check runs (the report is complete, not
        first-failure-only).
    """

    ok: bool
    errors: list[str]
    warnings: list[str]
    resolved_pipeline_preset: str
    expected_ids: dict
    checks: list["PreflightCheck"]

    def __bool__(self) -> bool:
        """Return ``True`` when the configuration is runnable (``ok``)."""
        return self.ok


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
        and ``checks`` (from the underlying :class:`PreflightReport`).
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


def assert_preset_compute_rows(bundle, *, with_artifact: bool = True) -> None:
    """Raise ``PreflightError`` if a preset's compute-time rows / sorter binary
    are missing.

    The mode-independent half of ``preflight_v2_pipeline``: the preset's
    preprocessing / (optional) artifact / sorter / display-analyzer-waveform
    Lookup rows plus the sorter binary/runtime. ``run_v2_pipeline``'s concat
    preflight calls this (``with_artifact=False`` -- a concat sort runs no
    artifact stage) so a concat run fails fast on a missing row instead of deep
    in the member/concat populate, matching the single-session preflight. The
    param-row existence queries mirror ``preflight_v2_pipeline``'s (kept in its
    report-building form there); the sorter binary/runtime check reuses the
    shared :func:`_check_local_sorter_runtime` via a raise-style adapter.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._params.sorter import (
        validate_execution_params,
    )
    from spyglass.spikesorting.v2._recipe_catalog import (
        waveform_params_for_preprocessing,
    )
    from spyglass.spikesorting.v2._sorting_dispatch import (
        MATLAB_SORTERS,
        matlab_container_required_message,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import (
        AnalyzerWaveformParameters,
        SorterParameters,
    )

    if not (
        PreprocessingParameters
        & {"preprocessing_params_name": bundle.preprocessing_params_name}
    ):
        raise PreflightError(
            "run_v2_pipeline: PreprocessingParameters row "
            f"{bundle.preprocessing_params_name!r} is missing. Run "
            "initialize_v2_defaults()."
        )
    if with_artifact and bundle.artifact_detection_params_name is not None:
        if not (
            ArtifactDetectionParameters
            & {
                "artifact_detection_params_name": (
                    bundle.artifact_detection_params_name
                )
            }
        ):
            raise PreflightError(
                "run_v2_pipeline: ArtifactDetectionParameters row "
                f"{bundle.artifact_detection_params_name!r} is missing. Run "
                "initialize_v2_defaults()."
            )
    sorter_params_query = SorterParameters & {
        "sorter": bundle.sorter,
        "sorter_params_name": bundle.sorter_params_name,
    }
    if not sorter_params_query:
        raise PreflightError(
            "run_v2_pipeline: SorterParameters row (sorter="
            f"{bundle.sorter!r}, sorter_params_name="
            f"{bundle.sorter_params_name!r}) is missing. Run "
            "initialize_v2_defaults()."
        )
    display_waveform_params_name = waveform_params_for_preprocessing(
        bundle.preprocessing_params_name
    )[0]
    if not (
        AnalyzerWaveformParameters
        & {"waveform_params_name": display_waveform_params_name}
    ):
        raise PreflightError(
            "run_v2_pipeline: AnalyzerWaveformParameters row "
            f"{display_waveform_params_name!r} (the display analyzer recipe for "
            f"preprocessing {bundle.preprocessing_params_name!r}) is missing. "
            "Run initialize_v2_defaults()."
        )

    def _raise_check(name, ok, fix):
        if not bool(ok):
            raise PreflightError(fix)
        return True

    # Sorter binary/runtime, dispatched on the execution backend exactly like
    # the single-session preflight: a LOCAL backend needs the sorter installed
    # here; a CONTAINER backend needs the container runtime (the sorter runtime
    # lives in the image, so a missing LOCAL install is irrelevant, but a
    # missing container runtime is a blocking failure -- preflight never falls
    # back to local execution).
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
            bundle, sis, SorterParameters._NON_SI_SORTERS, _raise_check
        )
    else:
        if execution_backend == "docker":
            runtime_ok, runtime_detail = _docker_runtime_available()
        else:
            runtime_ok, runtime_detail = _singularity_runtime_available()
        if not runtime_ok:
            raise PreflightError(
                f"run_v2_pipeline: the {execution_backend} execution backend "
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
) -> list[str]:
    """Raise ``PreflightError`` if a concat run's prerequisites are missing.

    Concat counterpart of :func:`preflight_v2_pipeline`: the group, its members,
    and -- because each member is sorted through the same single-session
    ``Recording`` build -- the per-member ``Raw`` / ``'raw data valid times'`` /
    sort-group-electrode / sampling-rate prerequisites, plus the preset's
    motion-correction row, the ``auto_curate`` metric/rule/metric-waveform rows
    (when opted in), and the compute-time param rows + sorter binary via
    :func:`assert_preset_compute_rows` (``with_artifact=False`` -- a concat sort
    runs no artifact stage). Fails fast BEFORE the heavy member / concat populate
    rather than deep in a member ``populate`` with an opaque error. Returns the
    advisory-warning list (currently always empty) for symmetry with
    :func:`preflight_v2_pipeline`.
    """
    from spyglass.common import IntervalList, Raw
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.recording import SortGroupV2
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

    # Each member is sorted through the same single-session Recording build, so
    # mirror preflight_v2_pipeline's per-session prerequisites for EVERY member;
    # SessionGroup.Member's FK set validates only that the master rows exist, not
    # Raw / 'raw data valid times' / a non-empty sort group / a matching rate, so
    # a partially-ingested or empty-sort-group member would otherwise fail deep
    # in the member populate with an opaque error.
    members = (SessionGroup.Member & group_key).fetch(
        "member_index", "nwb_file_name", "sort_group_id", as_dict=True
    )
    for member in members:
        nwb = member["nwb_file_name"]
        sort_group_id = int(member["sort_group_id"])
        tag = (
            f"concat member {member['member_index']} "
            f"({nwb!r}, sort_group_id={sort_group_id})"
        )
        if not (Raw & {"nwb_file_name": nwb}):
            raise PreflightError(
                f"run_v2_pipeline: {tag} has no Raw electrical-series row (the "
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
                f"run_v2_pipeline: {tag} is missing IntervalList 'raw data "
                "valid times', which the recording build reads for the raw "
                "sample bounds. Re-run ingestion."
            )
        if not (
            SortGroupV2.SortGroupElectrode
            & {"nwb_file_name": nwb, "sort_group_id": sort_group_id}
        ):
            raise PreflightError(
                f"run_v2_pipeline: {tag} SortGroupV2 has zero electrode "
                "members; Recording.populate would raise 'has zero electrodes'. "
                "Recreate it with SortGroupV2.set_group_by_shank(nwb_file_name="
                "...)."
            )
        if bundle.sampling_rate_hz is not None:
            actual_rate = float(
                (Raw & {"nwb_file_name": nwb}).fetch1("sampling_rate")
            )
            if (
                abs(actual_rate - bundle.sampling_rate_hz)
                > 0.005 * bundle.sampling_rate_hz
            ):
                raise PreflightError(
                    f"run_v2_pipeline: {tag} samples at {actual_rate:g} Hz but "
                    f"the concat preset is tuned for {bundle.sampling_rate_hz} "
                    "Hz (the rate-keyed sorter row "
                    f"{bundle.sorter_params_name!r} holds its clip_size / "
                    "detect_interval snippet window at that rate). Every member "
                    "must share the preset's acquisition rate."
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

    # Auto-curation prerequisites (preset-level, same rows as single-session),
    # only when the caller opts into auto_curate -- so a concat auto-curate run
    # fails fast on a missing metric / rule / metric-waveform recipe rather than
    # after the member + concat + sort compute.
    if auto_curate:
        from spyglass.spikesorting.v2._recipe_catalog import (
            waveform_params_for_preprocessing,
        )
        from spyglass.spikesorting.v2.metric_curation import (
            AutoCurationRules,
            QualityMetricParameters,
        )
        from spyglass.spikesorting.v2.sorting import (
            AnalyzerWaveformParameters,
        )

        if not (
            QualityMetricParameters
            & {"metric_params_name": bundle.metric_params_name}
        ):
            raise PreflightError(
                "run_v2_pipeline: QualityMetricParameters row "
                f"{bundle.metric_params_name!r} (the auto-curation metric set) "
                "is missing. Run initialize_v2_defaults()."
            )
        if not (
            AutoCurationRules
            & {"auto_curation_rules_name": bundle.auto_curation_rules_name}
        ):
            raise PreflightError(
                "run_v2_pipeline: AutoCurationRules row "
                f"{bundle.auto_curation_rules_name!r} (the auto-curation rule "
                "set) is missing. Run initialize_v2_defaults()."
            )
        metric_waveform_params_name = waveform_params_for_preprocessing(
            bundle.preprocessing_params_name
        )[1]
        if not (
            AnalyzerWaveformParameters
            & {"waveform_params_name": metric_waveform_params_name}
        ):
            raise PreflightError(
                "run_v2_pipeline: AnalyzerWaveformParameters row "
                f"{metric_waveform_params_name!r} (the whitened metric analyzer "
                "recipe auto-curation scores on) is missing. Run "
                "initialize_v2_defaults()."
            )

    assert_preset_compute_rows(bundle, with_artifact=False)
    return []


def preflight_v2_pipeline(
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    pipeline_preset: str = DEFAULT_PIPELINE_PRESET,
    auto_curate: bool = False,
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
    Two cases short-circuit before any database access, because the remaining
    checks would be meaningless: an unknown ``pipeline_preset`` (the later
    checks need the resolved param names), and a motion-pinned
    (concatenated-group) preset, which ``run_v2_pipeline``'s single-session
    inputs cannot run.

    Parameters
    ----------
    nwb_file_name, sort_group_id, interval_list_name, team_name, pipeline_preset
        The same inputs as :func:`run_v2_pipeline`.
    auto_curate
        Match ``run_v2_pipeline(auto_curate=...)``. When True, also verify the
        auto-curation prerequisites (the preset's ``QualityMetricParameters`` /
        ``AutoCurationRules`` rows and the whitened metric analyzer recipe), so
        a missing one fails preflight rather than after the upstream compute.

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

    # 1. pipeline_preset_known. Short-circuit before any DB access on failure: the
    # remaining checks (and expected_ids) all derive from the resolved
    # bundle's param names, which are unknown for a bogus pipeline preset.
    if pipeline_preset not in _PIPELINE_PRESETS:
        _check(
            "pipeline_preset_known",
            False,
            f"unknown pipeline_preset {pipeline_preset!r}. Available pipeline presets: "
            f"{sorted(_PIPELINE_PRESETS)}. Call describe_pipeline_presets() to see what each "
            "one does.",
        )
        return PreflightReport(
            ok=False,
            errors=[c.fix for c in checks if not c.ok],
            warnings=warnings,
            resolved_pipeline_preset=pipeline_preset,
            expected_ids={},
            checks=checks,
        )
    _check("pipeline_preset_known", True, "")
    bundle = _PIPELINE_PRESETS[pipeline_preset]

    # A motion-pinned preset targets a concatenated session group (motion
    # correction runs on the ConcatenatedRecording path, not single-session
    # Recording), which run_v2_pipeline's single-session inputs cannot drive
    # (concat mode handles it). Short-circuit BEFORE the SpikeInterface /
    # DB-touching checks (like the unknown-preset case above) so this verdict
    # stays database-free.
    if not _check(
        "single_session_preset",
        bundle.motion_correction_params_name is None,
        f"pipeline_preset {pipeline_preset!r} pins motion correction "
        f"(motion_correction_params_name={bundle.motion_correction_params_name!r}), "
        "so it targets a concatenated session group, not the single-session "
        "inputs given. Run it in concat mode (concat_session_group_owner / "
        "concat_session_group_name), or choose a non-concat preset.",
    ):
        return PreflightReport(
            ok=False,
            errors=[c.fix for c in checks if not c.ok],
            warnings=warnings,
            resolved_pipeline_preset=pipeline_preset,
            expected_ids={},
            checks=checks,
        )

    import spikeinterface.sorters as sis

    from spyglass.common import IntervalList, LabTeam, Raw, Session
    from spyglass.spikesorting.v2._selection_identity import (
        artifact_detection_identity_payload,
        deterministic_id,
        recording_identity_payload,
        sorting_identity_payload,
    )
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
        resolve_recording_input_hash,
    )
    from spyglass.spikesorting.v2._recipe_catalog import (
        waveform_params_for_preprocessing,
    )
    from spyglass.spikesorting.v2.sorting import (
        AnalyzerWaveformParameters,
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    if auto_curate:
        from spyglass.spikesorting.v2.metric_curation import (
            AutoCurationRules,
            QualityMetricParameters,
        )

    sort_group_id = int(sort_group_id)

    # 2-6. Upstream session/raw/interval/team/sort-group rows.
    _check(
        "session_exists",
        Session & {"nwb_file_name": nwb_file_name},
        f"session {nwb_file_name!r} is not ingested. Ingest it with "
        "insert_sessions(...) first.",
    )
    # ``RecordingSelection`` FKs ``Raw`` (not ``Session``), so a session whose
    # ``Raw`` row is missing (e.g. a partial ingestion) would pass
    # ``session_exists`` yet fail the recording insert with an opaque
    # foreign-key error. Check ``Raw`` explicitly so preflight stays honest.
    _check(
        "raw_exists",
        Raw & {"nwb_file_name": nwb_file_name},
        f"Raw electrical-series row for {nwb_file_name!r} is missing (the "
        "session is ingested but its Raw data is not). Re-run ingestion "
        "(e.g. populate_all_common / insert_sessions) so Raw is populated.",
    )
    _check(
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
    _check(
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
    _check(
        "team_exists",
        LabTeam & {"team_name": team_name},
        f"LabTeam {team_name!r} does not exist. Create it with "
        "LabTeam.insert1({'team_name': ..., 'team_description': ...}).",
    )
    sort_group_exists = _check(
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
        _check(
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

    # 6-8. The preset's parameter Lookup rows.
    preprocessing_params_exist = _check(
        "preprocessing_params_exist",
        PreprocessingParameters
        & {"preprocessing_params_name": bundle.preprocessing_params_name},
        f"PreprocessingParameters row {bundle.preprocessing_params_name!r} is "
        "missing. Run initialize_v2_defaults().",
    )
    # A None artifact name means the preset runs no artifact stage (skip /
    # concat), so there is no ArtifactDetectionParameters row to require.
    if bundle.artifact_detection_params_name is not None:
        _check(
            "artifact_detection_params_exist",
            ArtifactDetectionParameters
            & {
                "artifact_detection_params_name": bundle.artifact_detection_params_name
            },
            f"ArtifactDetectionParameters row {bundle.artifact_detection_params_name!r} "
            "is missing. Run initialize_v2_defaults().",
        )
    sorter_params_query = SorterParameters & {
        "sorter": bundle.sorter,
        "sorter_params_name": bundle.sorter_params_name,
    }
    sorter_params_exist = _check(
        "sorter_params_exist",
        sorter_params_query,
        f"SorterParameters row (sorter={bundle.sorter!r}, "
        f"sorter_params_name={bundle.sorter_params_name!r}) is missing. "
        "Run initialize_v2_defaults().",
    )
    # The display analyzer recipe is region-resolved from the preprocessing
    # recipe and FK-required on the Sorting row, so Sorting.make_fetch fails if
    # its AnalyzerWaveformParameters row is missing. Gate it here (up front)
    # rather than crashing deep in populate, matching the other params checks.
    display_waveform_params_name = waveform_params_for_preprocessing(
        bundle.preprocessing_params_name
    )[0]
    _check(
        "analyzer_waveform_params_exist",
        AnalyzerWaveformParameters
        & {"waveform_params_name": display_waveform_params_name},
        f"AnalyzerWaveformParameters row {display_waveform_params_name!r} "
        "(the display analyzer recipe for preprocessing "
        f"{bundle.preprocessing_params_name!r}) is missing. Run "
        "initialize_v2_defaults().",
    )

    # 8a. auto-curation prerequisites -- only when the caller opts into
    # auto_curate, so a default run is unchanged. CurationEvaluation scores the
    # root curation with the preset's metric + auto-curation rule rows on the
    # whitened (metric) analyzer recipe, so a missing one of those would
    # otherwise fail only after the upstream compute. The metric waveform row is
    # the [1] element of the same source-resolved (display, metric) pair.
    if auto_curate:
        _check(
            "metric_params_exist",
            QualityMetricParameters
            & {"metric_params_name": bundle.metric_params_name},
            f"QualityMetricParameters row {bundle.metric_params_name!r} (the "
            "auto-curation metric set) is missing. Run "
            "initialize_v2_defaults().",
        )
        _check(
            "auto_curation_rules_exist",
            AutoCurationRules
            & {"auto_curation_rules_name": bundle.auto_curation_rules_name},
            f"AutoCurationRules row {bundle.auto_curation_rules_name!r} (the "
            "auto-curation rule set) is missing. Run initialize_v2_defaults().",
        )
        metric_waveform_params_name = waveform_params_for_preprocessing(
            bundle.preprocessing_params_name
        )[1]
        _check(
            "metric_waveform_params_exist",
            AnalyzerWaveformParameters
            & {"waveform_params_name": metric_waveform_params_name},
            f"AnalyzerWaveformParameters row {metric_waveform_params_name!r} "
            "(the whitened metric analyzer recipe auto-curation scores on) is "
            "missing. Run initialize_v2_defaults().",
        )

    # 8b. sampling_rate_matches. The MS4/MS5 snippet window (clip_size /
    # detect_interval on the rate-keyed sorter row) assumes a specific
    # acquisition rate, so a 30 kHz preset on a 20 kHz recording (or the
    # reverse) silently sorts with a mistuned window. The clusterless preset
    # is rate-agnostic (sampling_rate_hz is None) and is skipped; the check is
    # also skipped if Raw is not ingested yet (raw_exists already reports that,
    # so this would only add a confusing second failure).
    if bundle.sampling_rate_hz is not None:
        raw = Raw & {"nwb_file_name": nwb_file_name}
        if raw:
            actual_rate = float(raw.fetch1("sampling_rate"))
            # 0.5% tolerance absorbs float drift in an estimated rate.
            rate_ok = (
                abs(actual_rate - bundle.sampling_rate_hz)
                <= 0.005 * bundle.sampling_rate_hz
            )
            _check(
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

    # 9. Sorter execution. The selected backend (local vs container) is read
    # ONLY from the SorterParameters row's execution_params (the single source of
    # truth -- never the preset). When the row is absent, sorter_params_exist
    # above already reported the blocking error and the backend is unknowable, so
    # the container / MATLAB-policy checks are skipped; the sorter NAME is still
    # validated as a local sorter (the pre-execution-params behavior) so a
    # misspelled sorter keeps its spelling hint.
    if not sorter_params_exist:
        _check_local_sorter_runtime(
            bundle, sis, SorterParameters._NON_SI_SORTERS, _check
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
                _check(
                    "sorter_execution_backend",
                    False,
                    matlab_container_required_message(bundle.sorter),
                )
            else:
                _check_local_sorter_runtime(
                    bundle, sis, SorterParameters._NON_SI_SORTERS, _check
                )
        else:
            # Container backend: verify the container RUNTIME (engine + Python
            # package), not the local sorter install. The sorter runtime lives
            # in the image, so a missing LOCAL runtime is irrelevant here. A
            # missing CONTAINER runtime is an actionable, blocking
            # selected-preset error -- preflight never silently falls back to
            # local execution.
            if execution_backend == "docker":
                runtime_ok, runtime_detail = _docker_runtime_available()
            else:
                runtime_ok, runtime_detail = _singularity_runtime_available()
            _check(
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

    # Non-blocking advisory: the "none" artifact params are a no-op
    # pass-through (no masking). "default" performs real amplitude-threshold
    # detection and is the legitimate built-in choice, so it is NOT warned.
    if bundle.artifact_detection_params_name == "none":
        warnings.append(
            "artifact_detection_params_name='none': no artifact masking will be "
            "applied for this run."
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
        recording_id = deterministic_id(
            "recording",
            {
                **recording_identity_payload(
                    {
                        "nwb_file_name": nwb_file_name,
                        "sort_group_id": sort_group_id,
                        "interval_list_name": interval_list_name,
                        "preprocessing_params_name": (
                            bundle.preprocessing_params_name
                        ),
                        "team_name": team_name,
                    }
                ),
                "recording_input_hash": resolve_recording_input_hash(
                    nwb_file_name,
                    sort_group_id,
                    bundle.preprocessing_params_name,
                ),
            },
        )
        # A None artifact name means no artifact-detection pass: the sort's
        # identity carries artifact_detection_id=None (matching
        # build_sorting_selection_plan), so the expected sorting_id derives from
        # None and there is no ArtifactDetectionSelection PK to expect.
        if bundle.artifact_detection_params_name is None:
            artifact_detection_id = None
        else:
            artifact_detection_id = deterministic_id(
                "artifact_detection",
                artifact_detection_identity_payload(
                    artifact_detection_params_name=bundle.artifact_detection_params_name,
                    recording_id=recording_id,
                ),
            )
        sorting_id = deterministic_id(
            "sorting",
            sorting_identity_payload(
                recording_id=recording_id,
                sorter=bundle.sorter,
                sorter_params_name=bundle.sorter_params_name,
                artifact_detection_id=artifact_detection_id,
            ),
        )
        # Per stage, ``exists`` is whether the SELECTION row exists (the run
        # would reuse this PK) and ``computed_exists`` whether the COMPUTED
        # output row exists (the populate already ran -- a reused, near-zero-cost
        # stage). Distinguishing them tells a caller what work the run would
        # actually do: a selection can exist with its output not yet populated.
        expected_ids = {
            "recording_id": {
                "id": recording_id,
                "exists": bool(
                    RecordingSelection & {"recording_id": recording_id}
                ),
                "computed_exists": bool(
                    Recording & {"recording_id": recording_id}
                ),
            },
            "artifact_detection_id": (
                {"id": None, "exists": False, "computed_exists": False}
                if artifact_detection_id is None
                else {
                    "id": artifact_detection_id,
                    "exists": bool(
                        ArtifactDetectionSelection
                        & {"artifact_detection_id": artifact_detection_id}
                    ),
                    "computed_exists": bool(
                        ArtifactDetection
                        & {"artifact_detection_id": artifact_detection_id}
                    ),
                }
            ),
            "sorting_id": {
                "id": sorting_id,
                "exists": bool(SortingSelection & {"sorting_id": sorting_id}),
                "computed_exists": bool(Sorting & {"sorting_id": sorting_id}),
            },
        }

    errors = [c.fix for c in checks if not c.ok]
    return PreflightReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        resolved_pipeline_preset=pipeline_preset,
        expected_ids=expected_ids,
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
            f"{caller}: unknown pipeline_preset {pipeline_preset!r}. Available "
            f"pipeline presets: {sorted(_PIPELINE_PRESETS)}. Call "
            "describe_pipeline_presets() to see what each preset does."
        )
    # A motion-pinned preset targets a concatenated session group, which the
    # single-session session runner cannot drive. Reject it here -- before the
    # SortGroupV2 access below -- so the session helpers fail fast (database-free)
    # with the concat-preset message, matching the single-group helpers.
    bundle = _PIPELINE_PRESETS[pipeline_preset]
    if bundle.motion_correction_params_name is not None:
        raise PipelineInputError(
            f"{caller}: pipeline_preset {pipeline_preset!r} pins motion "
            "correction (motion_correction_params_name="
            f"{bundle.motion_correction_params_name!r}), so it targets a "
            "concatenated session group, not the single-session inputs the "
            "session runner drives. Concatenated (same-day) sorting is not "
            "available through the session runner; run it directly with "
            "run_v2_pipeline concat mode (concat_session_group_owner / "
            "concat_session_group_name), or choose a non-concat preset here."
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
        )
        group_reports.append(
            {
                "sort_group_id": sort_group_id,
                "ok": report.ok,
                "errors": report.errors,
                "warnings": report.warnings,
                "expected_ids": report.expected_ids,
                "checks": report.checks,
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
