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
        The selection PKs a subsequent ``run_v2_pipeline`` would produce,
        each annotated with whether the row already exists, e.g.
        ``{"recording_id": {"id": UUID(...), "exists": False}, ...}``. IDs
        are computed DB-free via ``deterministic_id``; ``exists`` is a
        ``& pk`` restriction. Empty when the preset is unknown (the param
        names needed to derive the IDs are then unavailable). For an ``ok``
        report each ``id`` equals the PK ``run_v2_pipeline`` returns.
        ``curation_id`` is intentionally excluded: it is assigned by
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


def preflight_v2_pipeline(
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    pipeline_preset: str = "franklab_tetrode_hippocampus_30khz_ms5_2026_06",
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

    Every check is a read-only restriction (``& {...}``) or a pure call;
    all checks run even after one fails, so the report lists every problem
    at once. An unknown ``pipeline_preset`` short-circuits before any database
    access (the later checks need the resolved param names).

    Parameters
    ----------
    nwb_file_name, sort_group_id, interval_list_name, team_name, pipeline_preset
        The same inputs as :func:`run_v2_pipeline`.

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

    import spikeinterface.sorters as sis

    from spyglass.common import IntervalList, LabTeam, Raw, Session
    from spyglass.spikesorting.v2._selection_identity import (
        artifact_detection_identity_payload,
        deterministic_id,
        recording_identity_payload,
        sorting_identity_payload,
    )
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
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
    _check(
        "preprocessing_params_exist",
        PreprocessingParameters
        & {"preprocessing_params_name": bundle.preprocessing_params_name},
        f"PreprocessingParameters row {bundle.preprocessing_params_name!r} is "
        "missing. Run initialize_v2_defaults().",
    )
    _check(
        "artifact_detection_params_exist",
        ArtifactDetectionParameters
        & {
            "artifact_detection_params_name": bundle.artifact_detection_params_name
        },
        f"ArtifactDetectionParameters row {bundle.artifact_detection_params_name!r} "
        "is missing. Run initialize_v2_defaults().",
    )
    _check(
        "sorter_params_exist",
        SorterParameters
        & {
            "sorter": bundle.sorter,
            "sorter_params_name": bundle.sorter_params_name,
        },
        f"SorterParameters row (sorter={bundle.sorter!r}, "
        f"sorter_params_name={bundle.sorter_params_name!r}) is missing. "
        "Run initialize_v2_defaults().",
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

    # 9. sorter_installed. Reuse the SAME strict gate insert_default uses:
    # the internal clusterless_thresholder (_NON_SI_SORTERS) is never an SI
    # binary and is always available; otherwise the sorter must be in
    # installed_sorters(), distinguishing "known but not installed" from
    # "misspelled / unknown" for the fix message.
    sorter_installed_ok = (
        bundle.sorter in SorterParameters._NON_SI_SORTERS
        or bundle.sorter in set(sis.installed_sorters())
    )
    if sorter_installed_ok:
        _check("sorter_installed", True, "")
    elif bundle.sorter in set(sis.available_sorters()):
        _check(
            "sorter_installed",
            False,
            f"sorter {bundle.sorter!r} is a known SpikeInterface sorter but "
            "its binary/runtime is not installed here "
            "(spikeinterface.sorters.installed_sorters()). Install it, or "
            "pick a preset whose sorter is installed.",
        )
    else:
        _check(
            "sorter_installed",
            False,
            f"sorter {bundle.sorter!r} is not a known SpikeInterface sorter "
            "(spikeinterface.sorters.available_sorters()) -- check the "
            "spelling or the preset.",
        )

    # 9b. sorter_runtime_available. installed_sorters() only checks that the SI
    # wrapper imports; for sorters that call a SEPARATE algorithm backend at run
    # time (see _SORTER_RUNTIME_BACKENDS) actually import that backend, so a
    # green sorter_installed cannot precede a sort-time failure -- whether the
    # backend is absent OR present-but-broken (e.g. a numpy<2-era ml_ms4alg
    # under the numpy>=2 baseline raising at import). Only runs when
    # sorter_installed passed: if the wrapper itself is missing, a second
    # "backend missing" failure would be contradictory ("listed as installed").
    backend_modules = _SORTER_RUNTIME_BACKENDS.get(bundle.sorter, ())
    if sorter_installed_ok and backend_modules:
        import importlib

        broken_backends = []
        for mod in backend_modules:
            try:
                importlib.import_module(mod)
            except Exception as exc:  # noqa: BLE001 - any import failure disqualifies
                broken_backends.append(f"{mod} ({type(exc).__name__}: {exc})")
        _check(
            "sorter_runtime_available",
            not broken_backends,
            f"sorter {bundle.sorter!r} is listed as installed but its runtime "
            "backend(s) cannot be imported, so the sort would crash: "
            f"{'; '.join(broken_backends)}. Install/repair the backend "
            "(mountainsort4 needs ml_ms4alg, which requires numpy<2), or pick a "
            "preset whose sorter runs in this environment (e.g. a MountainSort5 "
            "preset).",
        )

    # Non-blocking advisory: the "none" artifact params are a no-op
    # pass-through (no masking). "default" performs real amplitude-threshold
    # detection and is the legitimate built-in choice, so it is NOT warned.
    if bundle.artifact_detection_params_name == "none":
        warnings.append(
            "artifact_detection_params_name='none': no artifact masking will be "
            "applied for this run."
        )

    # expected_ids: the deterministic selection PKs this run would produce.
    # Pure hashes of (preset params + inputs) via the SAME payload builders
    # insert_selection uses, so they cannot drift; computable once the
    # preset resolves, regardless of whether the rows exist yet. ``exists``
    # is a read-only & pk check.
    recording_id = deterministic_id(
        "recording",
        recording_identity_payload(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
                "interval_list_name": interval_list_name,
                "preprocessing_params_name": bundle.preprocessing_params_name,
                "team_name": team_name,
            }
        ),
    )
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
    expected_ids = {
        "recording_id": {
            "id": recording_id,
            "exists": bool(RecordingSelection & {"recording_id": recording_id}),
        },
        "artifact_detection_id": {
            "id": artifact_detection_id,
            "exists": bool(
                ArtifactDetectionSelection
                & {"artifact_detection_id": artifact_detection_id}
            ),
        },
        "sorting_id": {
            "id": sorting_id,
            "exists": bool(SortingSelection & {"sorting_id": sorting_id}),
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
    pipeline_preset: "str | None",
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
    pipeline_preset: "str | None" = None,
    sort_group_ids: "list[int] | None" = None,
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
