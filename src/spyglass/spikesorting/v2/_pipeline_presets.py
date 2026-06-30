"""Pipeline presets: the dated franklab production recipe bundles.

Extracted verbatim from ``pipeline.py`` (behavior-preserving) to keep the
orchestration facade small. ``pipeline.py`` re-exports the public names
(``list_pipeline_presets``, ``describe_pipeline_presets``,
``describe_pipeline_preset``, ``describe_preset``) so notebook import paths
are unchanged.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from spyglass.spikesorting.v2._recipe_catalog import pipeline_preset_specs

if TYPE_CHECKING:
    import pandas as pd


class _PipelinePreset(BaseModel):
    """A v2 pipeline preset: a bundle of Lookup-row names.

    The orchestrator consults this bundle once, then drives the
    ``insert_selection`` -> ``populate`` chain on each stage. ``params``
    Lookup row names must exist in the database before
    ``run_v2_pipeline`` is called; the preset itself does NOT insert
    Lookup rows (default rows are inserted explicitly by callers via
    ``*.insert_default()``).

    The human-facing fields (``intended_use``, ``threshold_units``,
    ``notes``) carry no runtime behavior; they describe the preset for
    ``describe_pipeline_presets()`` so a scientist can choose one without reading
    module source. They default to ``""`` so external presets need not
    supply them, but the built-ins below populate them.
    """

    model_config = ConfigDict(extra="forbid")
    preprocessing_params_name: str
    sorter: str
    sorter_params_name: str
    # Curation-stage row names: the quality-metric set and the auto-curation
    # rule set the preset computes/suggests. Required so every preset declares
    # its curation; consumed by the orchestrator only when the caller opts into
    # auto-curation (otherwise a convenience run stays initial-curation-only).
    metric_params_name: str
    auto_curation_rules_name: str
    # artifact_detection_params_name is optional: None means the preset runs no
    # artifact detection (a skip-artifact single-session preset, or a concat
    # preset -- concat sorts carry no ArtifactDetectionSource row). A non-None
    # value names the ArtifactDetectionParameters row.
    artifact_detection_params_name: "str | None" = None
    # motion_correction_params_name is optional: None for ordinary single-
    # session presets (motion is selected per recording); a concat preset sets
    # it ("auto" resolves to rigid_fast for same-day session groups).
    motion_correction_params_name: "str | None" = None
    # Discovery metadata (no runtime behavior) -- the axes a scientist picks a
    # preset by. probe_type is informational: the recipe is set by target
    # region (the preproc high-pass) and sampling rate (the sorter window),
    # not by probe geometry, so e.g. the tetrode- and probe-hippocampus 30 kHz
    # presets resolve to the SAME parameter rows.
    probe_type: str = ""  # "tetrode" | "probe" | "neuropixels" | "" (n/a)
    target_region: str = ""  # "hippocampus" | "cortex" -> preproc high-pass
    sampling_rate_hz: "int | None" = None
    sorter_family: str = ""
    adjacency_radius_um: "float | None" = (
        None  # informational MS4 spatial radius
    )
    recommendation_status: str = ""  # production | alternative | experimental
    # NB: the sorter EXECUTION backend (local vs Docker/Singularity) is NOT a
    # preset field -- it lives on the referenced ``SorterParameters.execution_params``
    # row (the single source of truth). ``describe_pipeline_preset(name)`` reads it
    # back from the row; this DB-free preset only names the sorter row.
    intended_use: str = ""  # one-line "when to reach for this pipeline preset"
    threshold_units: str = ""  # detection-threshold units (sigma / µV)
    notes: str = ""  # key assumptions (probe geometry, sampling rate, etc.)


# The preset bundles + UX metadata are the single source of truth in
# ``_recipe_catalog.pipeline_preset_specs`` (alongside the parameter-row
# recipes they reference); build the validated ``_PipelinePreset`` objects
# from those plain-dict specs here.
_PIPELINE_PRESETS: dict[str, _PipelinePreset] = {
    name: _PipelinePreset(**spec)
    for name, spec in pipeline_preset_specs().items()
}


def list_pipeline_presets() -> list[str]:
    """Return the sorted pipeline-preset names accepted by ``run_v2_pipeline``.

    Notebook-discoverable accessor so users don't need to read the
    module source to learn what's available.

    Examples
    --------
    >>> from spyglass.spikesorting.v2.pipeline import list_pipeline_presets
    >>> any("ms4_2026_06" in p for p in list_pipeline_presets())
    True
    """
    return sorted(_PIPELINE_PRESETS)


def describe_pipeline_presets() -> "pd.DataFrame":
    """Return a table describing each pipeline preset accepted by ``run_v2_pipeline``.

    Companion to :func:`list_pipeline_presets` that adds the detail a scientist
    needs to choose a preset -- the sorter, the parameter-row names each
    stage uses, the intended use, and (a known footgun) the units of the
    detection threshold -- without reading module source. Pure and
    database-free: it reads only the in-module preset metadata and
    queries/inserts nothing. ``pandas`` is imported lazily so
    ``import spyglass.spikesorting.v2.pipeline`` stays cheap.

    Returns
    -------
    pandas.DataFrame
        One row per pipeline preset, sorted by name, with discovery
        metadata (``recommendation_status``, ``probe_type``,
        ``target_region``, ``sampling_rate_hz``, ``sorter``,
        ``sorter_family``, ``adjacency_radius_um``), the resolved row names
        (``preprocessing_params_name`` / ``artifact_detection_params_name``
        / ``sorter_params_name``), ``intended_use``, ``threshold_units``,
        and ``notes``. Call ``.to_dict("records")`` for raw rows.

    Examples
    --------
    >>> from spyglass.spikesorting.v2.pipeline import describe_pipeline_presets
    >>> "recommendation_status" in describe_pipeline_presets().columns
    True
    """
    import pandas as pd

    columns = [
        "pipeline_preset",
        "recommendation_status",
        "probe_type",
        "target_region",
        "sampling_rate_hz",
        "sorter",
        "sorter_family",
        "adjacency_radius_um",
        "preprocessing_params_name",
        "artifact_detection_params_name",
        "sorter_params_name",
        "metric_params_name",
        "auto_curation_rules_name",
        "motion_correction_params_name",
        "intended_use",
        "threshold_units",
        "notes",
    ]
    rows = [
        {
            "pipeline_preset": name,
            "recommendation_status": preset.recommendation_status,
            "probe_type": preset.probe_type,
            "target_region": preset.target_region,
            "sampling_rate_hz": preset.sampling_rate_hz,
            "sorter": preset.sorter,
            "sorter_family": preset.sorter_family,
            "adjacency_radius_um": preset.adjacency_radius_um,
            "preprocessing_params_name": preset.preprocessing_params_name,
            "artifact_detection_params_name": (
                preset.artifact_detection_params_name
            ),
            "sorter_params_name": preset.sorter_params_name,
            "metric_params_name": preset.metric_params_name,
            "auto_curation_rules_name": preset.auto_curation_rules_name,
            "motion_correction_params_name": (
                preset.motion_correction_params_name
            ),
            "intended_use": preset.intended_use,
            "threshold_units": preset.threshold_units,
            "notes": preset.notes,
        }
        for name, preset in sorted(_PIPELINE_PRESETS.items())
    ]
    return pd.DataFrame(rows, columns=columns)


_PRESET_DETAIL_COLUMNS = [
    "stage",
    "params_row_name",
    "key",
    "value",
    "params_schema_version",
    "job_kwargs",
]


def describe_pipeline_preset(name: str) -> "pd.DataFrame":
    """Unpack one pipeline preset into its full, validated parameter values.

    The singular companion to :func:`describe_pipeline_presets`: where the
    plural helper lists every preset and the parameter-row *names* each stage
    uses, this resolves ONE preset to the actual VALUES of its preprocessing,
    artifact-detection, sorter, metric, auto-curation, and (for concat presets)
    motion-correction parameter rows -- so you can see exactly what
    ``run_v2_pipeline(..., pipeline_preset=name)`` will do before running it.

    Unlike the DB-free :func:`describe_pipeline_presets`, this reads the live
    parameter Lookup tables. If a referenced row is missing it raises a clear
    "run ``initialize_v2_defaults()``" message rather than failing opaquely.

    Parameters
    ----------
    name : str
        A pipeline-preset name from :func:`list_pipeline_presets` /
        :func:`describe_pipeline_presets`.

    Returns
    -------
    pandas.DataFrame
        Long-format, one row per parameter, columns ``stage`` (``"preset"`` /
        ``"preprocessing"`` / ``"artifact_detection"`` / ``"motion_correction"``
        / ``"sorter"`` / ``"sorter_execution"`` / ``"metric"`` /
        ``"auto_curation"``), ``params_row_name``, ``key`` (dotted path into
        the validated blob), and ``value``. The ``"motion_correction"`` stage is
        present only for a concat preset (one that pins a motion row); the
        ``"metric"`` rows unpack the ``QualityMetricParameters`` columns
        (``metric_names`` / ``metric_kwargs`` / ``template_metric_columns`` /
        ``skip_pc_metrics``) and the ``"auto_curation"`` rows carry the
        ``AutoCurationRules`` master fields plus one ``rule.<index>`` entry per
        ordered label rule (``"<rule_name>: <metric> <op> <threshold> ->
        <label>"``). Parameter-row entries also carry
        ``params_schema_version`` and ``job_kwargs`` so the display shows the
        full row resolved by the preset, not just the core params blob. The
        ``"sorter_execution"`` rows surface the sorter's ``execution_params``
        backend/container provenance read from the ``SorterParameters`` row (the
        single source of truth -- the DB-free :func:`describe_pipeline_presets`
        does not carry execution fields); their own ``schema_version`` is one of
        the flattened keys, so ``params_schema_version`` is left blank for them.
        Preset entries include the human-facing ``threshold_units`` field so
        detection thresholds are never unitless.
    """
    import pandas as pd

    # Validate the name BEFORE importing the table modules: importing them
    # triggers DataJoint ``@schema`` decoration (a DB connection), so an unknown
    # name must reject first to keep that path database-free.
    if name not in _PIPELINE_PRESETS:
        raise ValueError(
            f"unknown pipeline_preset {name!r}. Available pipeline presets: "
            f"{sorted(_PIPELINE_PRESETS)}. Call describe_pipeline_presets() to "
            "see what each one does."
        )
    preset = _PIPELINE_PRESETS[name]

    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.session_group import (
        MotionCorrectionParameters,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters
    from spyglass.spikesorting.v2.utils import _jsonable_blob

    def _flatten(prefix, value):
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                key = f"{prefix}.{sub_key}" if prefix else str(sub_key)
                yield from _flatten(key, sub_value)
        else:
            yield prefix, value

    def _fetch_params(table, restriction, label):
        rel = table & restriction
        if not rel:
            raise ValueError(
                f"describe_pipeline_preset: {label} row {restriction!r} "
                f"referenced by preset {name!r} is not in the database. Run "
                "initialize_v2_defaults() to install the shipped parameter "
                "catalog."
            )
        params, params_schema_version, job_kwargs = rel.fetch1(
            "params", "params_schema_version", "job_kwargs"
        )
        return {
            "params": _jsonable_blob(params),
            "params_schema_version": int(params_schema_version),
            "job_kwargs": _jsonable_blob(job_kwargs),
        }

    def _detail_row(
        *,
        stage,
        params_row_name,
        key,
        value,
        params_schema_version=None,
        job_kwargs=None,
    ):
        return {
            "stage": stage,
            "params_row_name": params_row_name,
            "key": key,
            "value": value,
            "params_schema_version": params_schema_version,
            "job_kwargs": job_kwargs,
        }

    rows = [
        _detail_row(
            stage="preset",
            params_row_name=name,
            key=key,
            value=value,
        )
        for key, value in (
            ("sorter", preset.sorter),
            ("target_region", preset.target_region),
            ("sampling_rate_hz", preset.sampling_rate_hz),
            ("recommendation_status", preset.recommendation_status),
            ("metric_params_name", preset.metric_params_name),
            ("auto_curation_rules_name", preset.auto_curation_rules_name),
            (
                "motion_correction_params_name",
                preset.motion_correction_params_name,
            ),
            ("threshold_units", preset.threshold_units),
            ("intended_use", preset.intended_use),
            ("notes", preset.notes),
        )
    ]
    # Build the value-resolving stages. Artifact detection is optional: a None
    # name means the preset runs no artifact stage, so there is no row to unpack.
    stage_specs = [
        (
            "preprocessing",
            preset.preprocessing_params_name,
            _fetch_params(
                PreprocessingParameters,
                {"preprocessing_params_name": preset.preprocessing_params_name},
                "PreprocessingParameters",
            ),
        ),
    ]
    if preset.artifact_detection_params_name is not None:
        stage_specs.append(
            (
                "artifact_detection",
                preset.artifact_detection_params_name,
                _fetch_params(
                    ArtifactDetectionParameters,
                    {
                        "artifact_detection_params_name": (
                            preset.artifact_detection_params_name
                        )
                    },
                    "ArtifactDetectionParameters",
                ),
            )
        )
    # Motion correction is optional: only a concat preset pins one. Its row has
    # the same params / schema / job_kwargs shape as preprocessing, so it unpacks
    # through the shared helper.
    if preset.motion_correction_params_name is not None:
        stage_specs.append(
            (
                "motion_correction",
                preset.motion_correction_params_name,
                _fetch_params(
                    MotionCorrectionParameters,
                    {
                        "motion_correction_params_name": (
                            preset.motion_correction_params_name
                        )
                    },
                    "MotionCorrectionParameters",
                ),
            )
        )
    stage_specs.append(
        (
            "sorter",
            preset.sorter_params_name,
            _fetch_params(
                SorterParameters,
                {
                    "sorter": preset.sorter,
                    "sorter_params_name": preset.sorter_params_name,
                },
                "SorterParameters",
            ),
        )
    )
    for stage, row_name, row in stage_specs:
        for key, value in _flatten("", row["params"]):
            rows.append(
                _detail_row(
                    stage=stage,
                    params_row_name=row_name,
                    key=key,
                    value=value,
                    params_schema_version=row["params_schema_version"],
                    job_kwargs=row["job_kwargs"],
                )
            )

    # The sorter EXECUTION backend (local vs Docker/Singularity + container
    # install provenance) lives on the SorterParameters row, not the preset, so
    # surface it here from the single source of truth. The execution blob's own
    # ``schema_version`` is one of the flattened keys, so ``params_schema_version``
    # (the params-stage column) is left None for these rows rather than
    # overloaded. The sorter row's existence was already verified above.
    execution_params = (
        SorterParameters
        & {
            "sorter": preset.sorter,
            "sorter_params_name": preset.sorter_params_name,
        }
    ).fetch1("execution_params")
    for key, value in _flatten("", _jsonable_blob(execution_params)):
        rows.append(
            _detail_row(
                stage="sorter_execution",
                params_row_name=preset.sorter_params_name,
                key=key,
                value=value,
            )
        )

    # Curation recipe values. QualityMetricParameters stores named columns (not a
    # single params blob), so unpack them into a params-shaped dict and flatten.
    metric_rel = QualityMetricParameters & {
        "metric_params_name": preset.metric_params_name
    }
    if not metric_rel:
        raise ValueError(
            "describe_pipeline_preset: QualityMetricParameters row "
            f"{preset.metric_params_name!r} referenced by preset {name!r} is "
            "not in the database. Run initialize_v2_defaults() to install the "
            "shipped parameter catalog."
        )
    (
        metric_names,
        metric_kwargs,
        template_metric_columns,
        skip_pc_metrics,
        metric_schema_version,
        metric_job_kwargs,
    ) = metric_rel.fetch1(
        "metric_names",
        "metric_kwargs",
        "template_metric_columns",
        "skip_pc_metrics",
        "params_schema_version",
        "job_kwargs",
    )
    metric_params = {
        "metric_names": _jsonable_blob(metric_names),
        "metric_kwargs": _jsonable_blob(metric_kwargs),
        "template_metric_columns": _jsonable_blob(template_metric_columns),
        "skip_pc_metrics": bool(skip_pc_metrics),
    }
    for key, value in _flatten("", metric_params):
        rows.append(
            _detail_row(
                stage="metric",
                params_row_name=preset.metric_params_name,
                key=key,
                value=value,
                params_schema_version=int(metric_schema_version),
                job_kwargs=_jsonable_blob(metric_job_kwargs),
            )
        )

    # Auto-curation: the master row (merge preset + kwargs) plus one row per
    # ordered label rule from the Rule part, so the actual thresholds are visible
    # rather than hidden behind the row name.
    auto_rel = AutoCurationRules & {
        "auto_curation_rules_name": preset.auto_curation_rules_name
    }
    if not auto_rel:
        raise ValueError(
            "describe_pipeline_preset: AutoCurationRules row "
            f"{preset.auto_curation_rules_name!r} referenced by preset "
            f"{name!r} is not in the database. Run initialize_v2_defaults() to "
            "install the shipped parameter catalog."
        )
    (
        auto_merge_preset,
        auto_merge_kwargs,
        auto_schema_version,
        auto_job_kwargs,
    ) = auto_rel.fetch1(
        "auto_merge_preset",
        "auto_merge_kwargs",
        "params_schema_version",
        "job_kwargs",
    )
    auto_master = {
        "auto_merge_preset": auto_merge_preset,
        "auto_merge_kwargs": _jsonable_blob(auto_merge_kwargs),
    }
    for key, value in _flatten("", auto_master):
        rows.append(
            _detail_row(
                stage="auto_curation",
                params_row_name=preset.auto_curation_rules_name,
                key=key,
                value=value,
                params_schema_version=int(auto_schema_version),
                job_kwargs=_jsonable_blob(auto_job_kwargs),
            )
        )
    rule_rows = (
        AutoCurationRules.Rule
        & {"auto_curation_rules_name": preset.auto_curation_rules_name}
    ).fetch(order_by="rule_index", as_dict=True)
    for rule in rule_rows:
        rows.append(
            _detail_row(
                stage="auto_curation",
                params_row_name=preset.auto_curation_rules_name,
                key=f"rule.{rule['rule_index']}",
                value=(
                    f"{rule['rule_name']}: {rule['metric_name']} "
                    f"{rule['operator']} {rule['threshold']} -> {rule['label']}"
                ),
                params_schema_version=int(auto_schema_version),
            )
        )
    return pd.DataFrame(rows, columns=_PRESET_DETAIL_COLUMNS)


# Alias: shorter discovery name for the same helper.
describe_preset = describe_pipeline_preset


def _assert_preset_rows_exist(name: str, preset: "_PipelinePreset") -> None:
    """Raise ``ValueError`` if any Lookup row the preset references is absent.

    A preset is only usable if every parameter row it names already exists, so
    ``register_preset`` checks them up front rather than letting the orchestrator
    fail with an opaque FK error mid-populate. Names the missing row and the
    table so the fix is obvious.
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.session_group import (
        MotionCorrectionParameters,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters

    checks = [
        (
            PreprocessingParameters,
            {"preprocessing_params_name": preset.preprocessing_params_name},
            "PreprocessingParameters",
            preset.preprocessing_params_name,
        ),
        (
            SorterParameters,
            {
                "sorter": preset.sorter,
                "sorter_params_name": preset.sorter_params_name,
            },
            "SorterParameters",
            preset.sorter_params_name,
        ),
        (
            QualityMetricParameters,
            {"metric_params_name": preset.metric_params_name},
            "QualityMetricParameters",
            preset.metric_params_name,
        ),
        (
            AutoCurationRules,
            {"auto_curation_rules_name": preset.auto_curation_rules_name},
            "AutoCurationRules",
            preset.auto_curation_rules_name,
        ),
    ]
    # artifact detection is optional: a None name means the preset runs no
    # artifact stage (skip / concat), so there is no row to require.
    if preset.artifact_detection_params_name is not None:
        checks.append(
            (
                ArtifactDetectionParameters,
                {
                    "artifact_detection_params_name": (
                        preset.artifact_detection_params_name
                    )
                },
                "ArtifactDetectionParameters",
                preset.artifact_detection_params_name,
            )
        )
    # motion correction is optional: only presets that pin one (e.g. concat
    # presets) carry it.
    if preset.motion_correction_params_name is not None:
        checks.append(
            (
                MotionCorrectionParameters,
                {
                    "motion_correction_params_name": (
                        preset.motion_correction_params_name
                    )
                },
                "MotionCorrectionParameters",
                preset.motion_correction_params_name,
            )
        )
    for table, restriction, label, row_name in checks:
        if not (table & restriction):
            raise ValueError(
                f"register_preset({name!r}): row {row_name!r} not found in "
                f"{label}. Insert it (e.g. via initialize_v2_defaults() or the "
                "table's insert) before registering the preset."
            )


def register_preset(name: str, preset, *, validate_rows: bool = True) -> str:
    """Register a custom pipeline preset at runtime; return its name.

    The public, source-free way for a lab to add a preset: external labs use the
    same dated ``{lab}_..._{date}`` naming as the built-ins. The preset is
    Pydantic-validated (unknown fields are rejected) and, by default, every
    parameter row it references is verified to exist so a typo fails here with a
    clear message rather than mid-populate.

    Parameters
    ----------
    name : str
        Registry name. Must not already exist (refuses to silently overwrite a
        built-in or a prior registration).
    preset : dict or _PipelinePreset
        The preset bundle (a dict is validated into a ``_PipelinePreset``).
    validate_rows : bool, optional
        If True (default), verify the referenced Lookup rows exist (requires a
        database connection). Pass False to register without the DB check.

    Returns
    -------
    str
        The registered ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not a non-empty string, already exists, the preset fails
        Pydantic validation, or (when ``validate_rows``) a referenced parameter
        row is missing.
    """
    # Guard the registry key first: a non-string name breaks list_pipeline_presets
    # (it sorts mixed-type keys) and an empty/blank name is unusable.
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"register_preset: name must be a non-empty string, got {name!r}."
        )
    if name in _PIPELINE_PRESETS:
        raise ValueError(
            f"pipeline preset {name!r} is already registered. Choose a fresh "
            f"name; existing presets: {sorted(_PIPELINE_PRESETS)}."
        )
    validated = (
        preset
        if isinstance(preset, _PipelinePreset)
        else _PipelinePreset(**preset)
    )
    if validate_rows:
        _assert_preset_rows_exist(name, validated)
    _PIPELINE_PRESETS[name] = validated
    return name


def _path_exists(blob, parts: list[str]) -> bool:
    """Return whether the dotted ``parts`` resolve to a key in ``blob``."""
    node = blob
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _deep_set(blob: dict, dotted_key: str, value) -> None:
    """Replace the value at ``dotted_key`` (known to exist) in ``blob``."""
    parts = dotted_key.split(".")
    node = blob
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def clone_preset(
    base_name: str,
    new_name: str,
    *,
    allow_duplicate_params: bool = False,
    **overrides,
) -> str:
    """Clone a pipeline preset with one or more parameter values changed.

    The "tune one knob" path: a user who wants a single non-default value need
    not hand-write a full Pydantic params dict. ``clone_preset`` resolves
    ``base_name`` to its preprocessing / artifact-detection / sorter parameter
    rows, applies ``overrides`` to the relevant stage's ``params`` blob, inserts
    the derived parameter rows under ``new_name``, and registers ``new_name`` via
    :func:`register_preset`. Inspect the result with
    :func:`describe_pipeline_preset`.

    Each override key is a dotted path into a stage's scientific ``params`` blob
    -- e.g. ``detect_threshold=4.0`` (a flat sorter knob, passable as a keyword)
    or ``**{"bandpass_filter.freq_min": 700.0}`` (a nested preprocessing knob;
    dotted keys are not valid Python identifiers, so pass them via ``**{...}``).
    The override is routed to whichever stage's blob already contains that path;
    a key that is not present in any stage blob is rejected rather than silently
    added. Only the scientific ``params`` blob is editable -- ``job_kwargs`` and
    the sorter's ``execution_params`` are carried over from the base row
    verbatim; change those by inserting a parameter row directly.

    Stages the overrides do not touch are not forked: the clone reuses the base
    preset's existing row names for them, so a one-knob change adds exactly one
    derived parameter row.

    The clone never mutates the base preset or its parameter rows -- it only adds
    new rows and a new registration. The derived rows are named ``new_name``, so
    re-running ``clone_preset`` with identical overrides (e.g. on a fresh process,
    where the in-memory registry is empty but the DB rows persist) is a no-op
    rather than a fork.

    Parameters
    ----------
    base_name : str
        A registered pipeline-preset name (see :func:`list_pipeline_presets`).
    new_name : str
        The fresh name for the clone. Must be a non-empty string not already in
        the preset registry, and is also used to name each derived parameter row.
    allow_duplicate_params : bool, optional
        If True, opt out of the duplicate-content guard when a derived row's
        content matches an existing row under a different name (see
        :func:`register_preset` / the parameter Lookups). Default False refuses
        to fork provenance, mirroring the parameter tables.
    **overrides
        Dotted ``stage_blob`` paths mapped to their new values.

    Returns
    -------
    str
        The registered ``new_name``.

    Raises
    ------
    ValueError
        If ``base_name`` is unknown; ``new_name`` is not a fresh non-empty
        string; no overrides are given; an override key matches no stage param
        (or is ambiguous across stages); an override value fails the stage's
        Pydantic schema; or a derived row name already exists with different
        content.
    DuplicateParameterContentError
        If a derived row's content matches an existing row under a different
        name and ``allow_duplicate_params`` is False.
    """
    # --- cheap, database-free validation (mirrors register_preset / describe) ---
    # Validate before importing the table modules: importing them triggers
    # DataJoint ``@schema`` decoration (a DB connection), so these checks stay
    # database-free and fail fast before any row is touched.
    if base_name not in _PIPELINE_PRESETS:
        raise ValueError(
            f"unknown pipeline_preset {base_name!r}. Available pipeline "
            f"presets: {sorted(_PIPELINE_PRESETS)}. Call "
            "describe_pipeline_presets() to see what each one does."
        )
    if not isinstance(new_name, str) or not new_name.strip():
        raise ValueError(
            f"clone_preset: new_name must be a non-empty string, got "
            f"{new_name!r}."
        )
    if new_name in _PIPELINE_PRESETS:
        raise ValueError(
            f"pipeline preset {new_name!r} is already registered. Choose a "
            f"fresh name; existing presets: {sorted(_PIPELINE_PRESETS)}."
        )
    if not overrides:
        raise ValueError(
            "clone_preset requires at least one override (the knob to change); "
            "to register an alias of an existing preset under a new name use "
            "register_preset()."
        )
    base = _PIPELINE_PRESETS[base_name]

    from spyglass.spikesorting.v2._parameter_identity import (
        parameter_fingerprint,
    )
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )
    from spyglass.spikesorting.v2._params.sorter import _get_sorter_schema
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import SorterParameters
    from spyglass.spikesorting.v2.utils import (
        _jsonable_blob,
        transaction_or_noop,
    )

    # One descriptor per stage. ``sorter`` is the per-sorter dispatch key (and
    # the extra ``SorterParameters`` primary-key column / ``execution_params``
    # carrier); it is None for the single-key Lookups. The artifact-detection
    # stage is included only when the base preset runs one (a None artifact name
    # means no artifact stage -- a skip-artifact or concat preset -- so there is
    # no base row to fetch and no artifact key to override).
    stages = {
        "preprocessing": {
            "table": PreprocessingParameters,
            "table_name": "PreprocessingParameters",
            "pk_field": "preprocessing_params_name",
            "base_row_name": base.preprocessing_params_name,
            "schema_cls": PreprocessingParamsSchema,
            "sorter": None,
        },
        "sorter": {
            "table": SorterParameters,
            "table_name": "SorterParameters",
            "pk_field": "sorter_params_name",
            "base_row_name": base.sorter_params_name,
            "schema_cls": _get_sorter_schema(base.sorter),
            "sorter": base.sorter,
        },
    }
    if base.artifact_detection_params_name is not None:
        stages["artifact_detection"] = {
            "table": ArtifactDetectionParameters,
            "table_name": "ArtifactDetectionParameters",
            "pk_field": "artifact_detection_params_name",
            "base_row_name": base.artifact_detection_params_name,
            "schema_cls": ArtifactDetectionParamsSchema,
            "sorter": None,
        }

    def _base_restriction(stage):
        restriction = {stage["pk_field"]: stage["base_row_name"]}
        if stage["sorter"] is not None:
            restriction["sorter"] = stage["sorter"]
        return restriction

    # Fetch every base row's params (and the carried-over blobs) up front: the
    # routing step needs all three blobs to disambiguate an override, and a
    # missing row must fail with an actionable message, not an opaque fetch1.
    for stage in stages.values():
        rel = stage["table"] & _base_restriction(stage)
        if not rel:
            raise ValueError(
                f"clone_preset: {stage['table_name']} row "
                f"{stage['base_row_name']!r} referenced by preset "
                f"{base_name!r} is not in the database. Run "
                "initialize_v2_defaults() to install the shipped parameter "
                "catalog."
            )
        if stage["sorter"] is not None:
            params, psv, job_kwargs, exec_params, exec_psv = rel.fetch1(
                "params",
                "params_schema_version",
                "job_kwargs",
                "execution_params",
                "execution_params_schema_version",
            )
            stage["base_execution_params"] = _jsonable_blob(exec_params)
            stage["base_execution_params_schema_version"] = int(exec_psv)
        else:
            params, psv, job_kwargs = rel.fetch1(
                "params", "params_schema_version", "job_kwargs"
            )
        stage["base_params"] = _jsonable_blob(params)
        stage["base_params_schema_version"] = int(psv)
        stage["base_job_kwargs"] = _jsonable_blob(job_kwargs)

    # Route each override to the one stage whose params blob already contains
    # its dotted path. A path present in no stage is a typo / unsupported key; a
    # path present in more than one (e.g. a top-level key shared across stages)
    # cannot be disambiguated by a dotted key alone.
    stage_overrides: dict[str, dict] = {name: {} for name in stages}
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        matches = [
            name
            for name, stage in stages.items()
            if _path_exists(stage["base_params"], parts)
        ]
        if not matches:
            raise ValueError(
                f"clone_preset: override key {dotted_key!r} does not match any "
                "parameter in the base preset's preprocessing, "
                "artifact-detection, or sorter params. Call "
                f"describe_pipeline_preset({base_name!r}) to see the keys you "
                "can override."
            )
        if len(matches) > 1:
            raise ValueError(
                f"clone_preset: override key {dotted_key!r} is ambiguous -- it "
                f"matches multiple stages {sorted(matches)}. clone_preset "
                "cannot disambiguate a top-level key shared across stages; "
                "insert the parameter row directly for this change."
            )
        stage_overrides[matches[0]][dotted_key] = value

    touched = [name for name in stages if stage_overrides[name]]

    # Step 1 -- build + Pydantic-validate every derived blob BEFORE any insert,
    # so a bad override raises the same teaching error as a direct parameter
    # insert and leaves the database untouched.
    derived_params: dict[str, dict] = {}
    for name in touched:
        stage = stages[name]
        blob = copy.deepcopy(stage["base_params"])
        for dotted_key, value in stage_overrides[name].items():
            _deep_set(blob, dotted_key, value)
        derived_params[name] = (
            stage["schema_cls"].model_validate(blob).model_dump()
        )

    def _fingerprint(
        stage,
        *,
        params,
        params_schema_version,
        job_kwargs,
        execution_params,
        execution_params_schema_version,
    ):
        return parameter_fingerprint(
            stage["table_name"],
            params=_jsonable_blob(params),
            params_schema_version=int(params_schema_version),
            job_kwargs=_jsonable_blob(job_kwargs),
            sorter=stage["sorter"],
            execution_params=execution_params,
            execution_params_schema_version=execution_params_schema_version,
        )

    # Step 2 -- insert the derived rows atomically. For each touched stage,
    # reconcile against any existing row already named ``new_name``: identical
    # content is an idempotent no-op (a re-run reuses it), different content is a
    # name collision and refused. The duplicate-content guard inside the table
    # ``insert`` separately refuses a derived row whose content matches a
    # DIFFERENT existing name (unless ``allow_duplicate_params``).
    with transaction_or_noop(PreprocessingParameters.connection):
        for name in touched:
            stage = stages[name]
            exec_params = stage.get("base_execution_params")
            exec_psv = (
                stage.get("base_execution_params_schema_version")
                if stage["sorter"] is not None
                else None
            )
            derived_fp = _fingerprint(
                stage,
                params=derived_params[name],
                params_schema_version=stage["base_params_schema_version"],
                job_kwargs=stage["base_job_kwargs"],
                execution_params=exec_params,
                execution_params_schema_version=exec_psv,
            )

            new_restriction = {stage["pk_field"]: new_name}
            if stage["sorter"] is not None:
                new_restriction["sorter"] = stage["sorter"]
            existing = stage["table"] & new_restriction
            if existing:
                if stage["sorter"] is not None:
                    e_params, e_psv, e_jk, e_ep, e_epsv = existing.fetch1(
                        "params",
                        "params_schema_version",
                        "job_kwargs",
                        "execution_params",
                        "execution_params_schema_version",
                    )
                    existing_fp = _fingerprint(
                        stage,
                        params=e_params,
                        params_schema_version=e_psv,
                        job_kwargs=e_jk,
                        execution_params=_jsonable_blob(e_ep),
                        execution_params_schema_version=int(e_epsv),
                    )
                else:
                    e_params, e_psv, e_jk = existing.fetch1(
                        "params", "params_schema_version", "job_kwargs"
                    )
                    existing_fp = _fingerprint(
                        stage,
                        params=e_params,
                        params_schema_version=e_psv,
                        job_kwargs=e_jk,
                        execution_params=None,
                        execution_params_schema_version=None,
                    )
                if existing_fp == derived_fp:
                    continue  # idempotent: the derived row already exists
                raise ValueError(
                    f"clone_preset: a {stage['table_name']} row named "
                    f"{new_name!r} already exists with different content. "
                    "Choose a different new_name, or drop the existing row "
                    "before re-cloning."
                )

            row = {
                stage["pk_field"]: new_name,
                "params": derived_params[name],
                "job_kwargs": stage["base_job_kwargs"],
            }
            if stage["sorter"] is not None:
                row["sorter"] = stage["sorter"]
                row["execution_params"] = stage["base_execution_params"]
            stage["table"].insert1(
                row, allow_duplicate_params=allow_duplicate_params
            )

    derived_preset = base.model_copy(
        update={stages[name]["pk_field"]: new_name for name in touched}
    )
    register_preset(new_name, derived_preset, validate_rows=True)
    return new_name
