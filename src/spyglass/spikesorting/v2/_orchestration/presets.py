"""Pipeline presets: the dated franklab production recipe bundles.

Holds the preset registry, its ``list_*`` / ``describe_*`` views, and
``register_pipeline_preset`` / ``clone_pipeline_preset`` for user presets.
``pipeline.py`` re-exports the public names.
"""

from __future__ import annotations

import copy

from pydantic import BaseModel, ConfigDict
import pandas as pd

from spyglass.spikesorting.v2._core.recipe_catalog import (
    pipeline_preset_specs,
    waveform_params_for_preprocessing,
)
from spyglass.spikesorting.v2._core.lookup_validation import (
    _jsonable_blob,
    parameter_row_fingerprint,
)
from spyglass.spikesorting.v2._params.artifact_detection import (
    ArtifactDetectionParamsSchema,
)
from spyglass.spikesorting.v2._params.preprocessing import (
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2._params.sorter import _get_sorter_schema


class _PipelinePreset(BaseModel):
    """A v2 pipeline preset: a bundle of Lookup-row names.

    The named rows must exist before ``run_v2_pipeline``; the preset inserts
    none. Discovery fields (``probe_type`` through ``notes``) have no runtime
    behavior. The sorter execution backend lives on the ``SorterParameters``
    row, not here.
    """

    model_config = ConfigDict(extra="forbid")
    preprocessing_params_name: str
    sorter: str
    sorter_params_name: str
    # Used only when the caller opts into auto-curation.
    metric_params_name: str
    auto_curation_rules_name: str
    # None runs no artifact detection; concat applies it per member.
    artifact_detection_params_name: "str | None" = None
    # Region and sampling rate set the recipe; probe_type is informational.
    probe_type: str = ""  # "tetrode" | "probe" | "neuropixels" | "" (n/a)
    target_region: str = ""  # "hippocampus" | "cortex" -> preproc high-pass
    sampling_rate_hz: "int | None" = None
    sorter_family: str = ""
    adjacency_radius_um: "float | None" = (
        None  # informational MS4 spatial radius
    )
    recommendation_status: str = ""  # production | alternative | experimental
    intended_use: str = ""  # one-line "when to reach for this pipeline preset"
    threshold_units: str = ""  # detection-threshold units (sigma / µV)
    notes: str = ""  # key assumptions (probe geometry, sampling rate, etc.)


# Specs live in ``_core.recipe_catalog.pipeline_preset_specs``.
_PIPELINE_PRESETS: dict[str, _PipelinePreset] = {
    name: _PipelinePreset(**spec)
    for name, spec in pipeline_preset_specs().items()
}


def _unknown_pipeline_preset_message(
    name,
    *,
    caller: "str | None" = None,
    hint: str = "Call describe_pipeline_presets() to see what each one does.",
) -> str:
    """The message for a ``pipeline_preset`` name not in the registry.

    Prefixed with ``"{caller}: "`` when ``caller`` is given.
    """
    message = (
        f"unknown pipeline_preset {name!r}. Available pipeline presets: "
        f"{sorted(_PIPELINE_PRESETS)}. {hint}"
    )
    return message if caller is None else f"{caller}: {message}"


def _require_preset_row(table, restriction: dict, *, caller, row, preset_name):
    """Return ``table & restriction``; raise ``ValueError`` if it is empty.

    The message points at ``initialize_v2_defaults()``.
    """
    rel = table & restriction
    if not rel:
        raise ValueError(
            f"{caller}: {table.__name__} row {row!r} referenced by preset "
            f"{preset_name!r} is not in the database. Run "
            "initialize_v2_defaults() to install the shipped parameter "
            "catalog."
        )
    return rel


RECOMMENDATION_STATUS_MEANINGS: dict[str, str] = {
    "production": (
        "Frank-lab validated and recommended for real science -- the default "
        "choice for its probe / target region / sampling rate."
    ),
    "alternative": (
        "A sound, working substitute for when the production recipe does not "
        "fit -- e.g. MountainSort5 is an alternative to production MountainSort4. "
        "Fine to use; just "
        "not the lab's first pick for that probe/region."
    ),
    "experimental": (
        "Not yet validated on Frank-lab data. It runs, but inspect the output "
        "before relying on it (e.g. multi-day concatenation, Neuropixels "
        "Kilosort4)."
    ),
}


def list_pipeline_presets() -> list[str]:
    """Return the sorted pipeline-preset names accepted by ``run_v2_pipeline``.
    Examples
    --------
    >>> from spyglass.spikesorting.v2.pipeline import list_pipeline_presets
    >>> any("ms4_2026_06" in p for p in list_pipeline_presets())
    True
    """
    return sorted(_PIPELINE_PRESETS)


def describe_pipeline_presets() -> "pd.DataFrame":
    """Return a table describing each pipeline preset accepted by ``run_v2_pipeline``.

    Database-free. :func:`describe_recommendation_status` defines the
    ``recommendation_status`` values.

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
        "intended_use",
        "threshold_units",
        "notes",
    ]
    rows = [
        {"pipeline_preset": name, **preset.model_dump()}
        for name, preset in sorted(_PIPELINE_PRESETS.items())
    ]
    return pd.DataFrame(rows, columns=columns)


def describe_recommendation_status() -> "pd.DataFrame":
    """Explain the ``recommendation_status`` values from describe_pipeline_presets.

    Database-free.

    Returns
    -------
    pandas.DataFrame
        One row per status (columns ``recommendation_status``, ``meaning``),
        ordered ``production`` -> ``alternative`` -> ``experimental``.

    Examples
    --------
    >>> from spyglass.spikesorting.v2.pipeline import (
    ...     describe_recommendation_status,
    ... )
    >>> set(describe_recommendation_status()["recommendation_status"]) == {
    ...     "production", "alternative", "experimental"
    ... }
    True
    """
    return pd.DataFrame(
        [
            {"recommendation_status": status, "meaning": meaning}
            for status, meaning in RECOMMENDATION_STATUS_MEANINGS.items()
        ],
        columns=["recommendation_status", "meaning"],
    )


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

    Reads the live parameter rows. A missing row raises with a pointer to
    ``initialize_v2_defaults()``.

    Parameters
    ----------
    name : str
        A pipeline-preset name from :func:`list_pipeline_presets` /
        :func:`describe_pipeline_presets`.

    Returns
    -------
    pandas.DataFrame
        Long format, one row per parameter: ``stage`` (``"preset"`` /
        ``"preprocessing"`` / ``"artifact_detection"`` / ``"sorter"`` /
        ``"sorter_execution"`` / ``"metric"`` / ``"auto_curation"``),
        ``params_row_name``, ``key`` (dotted path), ``value``,
        ``params_schema_version`` and ``job_kwargs``. Each auto-curation rule
        is a ``rule.<index>`` entry. ``"sorter_execution"`` rows leave
        ``params_schema_version`` blank; their ``schema_version`` is a key.
    """
    # Before table imports, which connect to the database.
    if name not in _PIPELINE_PRESETS:
        raise ValueError(_unknown_pipeline_preset_message(name))
    preset = _PIPELINE_PRESETS[name]

    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import SorterParameters

    def _flatten(prefix, value):
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                key = f"{prefix}.{sub_key}" if prefix else str(sub_key)
                yield from _flatten(key, sub_value)
        else:
            yield prefix, value

    def _fetch_params(table, restriction):
        rel = _require_preset_row(
            table,
            restriction,
            caller="describe_pipeline_preset",
            row=restriction,
            preset_name=name,
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
            ("threshold_units", preset.threshold_units),
            ("intended_use", preset.intended_use),
            ("notes", preset.notes),
        )
    ]
    stage_specs = [
        (
            "preprocessing",
            preset.preprocessing_params_name,
            _fetch_params(
                PreprocessingParameters,
                {"preprocessing_params_name": preset.preprocessing_params_name},
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

    # QualityMetricParameters stores named columns, not one params blob.
    metric_rel = _require_preset_row(
        QualityMetricParameters,
        {"metric_params_name": preset.metric_params_name},
        caller="describe_pipeline_preset",
        row=preset.metric_params_name,
        preset_name=name,
    )
    (
        metric_names,
        metric_kwargs,
        template_metric_columns,
        skip_pc_metrics,
        observed_presence_bin_duration_s,
        metric_schema_version,
        metric_job_kwargs,
    ) = metric_rel.fetch1(
        "metric_names",
        "metric_kwargs",
        "template_metric_columns",
        "skip_pc_metrics",
        "observed_presence_bin_duration_s",
        "params_schema_version",
        "job_kwargs",
    )
    metric_params = {
        "metric_names": _jsonable_blob(metric_names),
        "metric_kwargs": _jsonable_blob(metric_kwargs),
        "template_metric_columns": _jsonable_blob(template_metric_columns),
        "skip_pc_metrics": bool(skip_pc_metrics),
        "observed_presence_bin_duration_s": float(
            observed_presence_bin_duration_s
        ),
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

    auto_rel = _require_preset_row(
        AutoCurationRules,
        {"auto_curation_rules_name": preset.auto_curation_rules_name},
        caller="describe_pipeline_preset",
        row=preset.auto_curation_rules_name,
        preset_name=name,
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


def _assert_preset_rows_exist(name: str, preset: "_PipelinePreset") -> None:
    """Raise ``ValueError`` if any Lookup row the preset references is absent.

    Fails up front rather than with a foreign-key error mid-populate.
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
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
    for table, restriction, label, row_name in checks:
        if not (table & restriction):
            raise ValueError(
                f"register_pipeline_preset({name!r}): row {row_name!r} not found in "
                f"{label}. Insert it (e.g. via initialize_v2_defaults() or the "
                "table's insert) before registering the preset."
            )


def register_pipeline_preset(
    name: str, preset, *, validate_rows: bool = True
) -> str:
    """Register a custom pipeline preset at runtime; return its name.

    Use the dated ``{lab}_..._{date}`` naming of the built-ins. The preset is
    validated and, by default, its referenced rows must exist.

    Parameters
    ----------
    name : str
        Registry name. Must not already exist.
    preset : dict or _PipelinePreset
        The preset bundle. The registry stores a freshly validated copy.
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
    # list_pipeline_presets sorts keys, so names must be strings.
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"register_pipeline_preset: name must be a non-empty string, got {name!r}."
        )
    if name in _PIPELINE_PRESETS:
        raise ValueError(
            f"pipeline preset {name!r} is already registered. Choose a fresh "
            f"name; existing presets: {sorted(_PIPELINE_PRESETS)}."
        )
    content = (
        preset.model_dump() if isinstance(preset, _PipelinePreset) else preset
    )
    validated = _PipelinePreset(**content)
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


def _derive_preset_parameters(stages, overrides, base_name):
    """Route overrides and validate all derived scientific blobs before writes."""
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
                f"clone_pipeline_preset: override key {dotted_key!r} does not match any "
                "parameter in the base preset's preprocessing, "
                "artifact-detection, or sorter params. Call "
                f"describe_pipeline_preset({base_name!r}) to see the keys you "
                "can override."
            )
        if len(matches) > 1:
            raise ValueError(
                f"clone_pipeline_preset: override key {dotted_key!r} is ambiguous -- it "
                f"matches multiple stages {sorted(matches)}. clone_pipeline_preset "
                "cannot disambiguate a top-level key shared across stages; "
                "insert the parameter row directly for this change."
            )
        stage_overrides[matches[0]][dotted_key] = value

    touched = [name for name in stages if stage_overrides[name]]

    # Validate every derived blob before any insert.
    derived_params: dict[str, dict] = {}
    for name in touched:
        stage = stages[name]
        blob = copy.deepcopy(stage["base_params"])
        for dotted_key, value in stage_overrides[name].items():
            _deep_set(blob, dotted_key, value)
        derived_params[name] = (
            stage["schema_cls"].model_validate(blob).model_dump()
        )

    return touched, derived_params


def clone_pipeline_preset(
    base_name: str,
    new_name: str,
    *,
    allow_duplicate_params: bool = False,
    **overrides,
) -> str:
    """Clone a pipeline preset with one or more parameter values changed.

    Inserts the changed stage rows under ``new_name`` and registers the
    preset; untouched stages reuse the base rows. Re-running with identical
    overrides reuses the rows.

    Each override key is a dotted path into one stage's ``params`` blob,
    e.g. ``detect_threshold=4.0`` or ``**{"bandpass_filter.freq_min": 700.0}``.
    Keys absent from every blob are rejected. ``job_kwargs`` and
    ``execution_params`` are copied from the base row.

    Parameters
    ----------
    base_name : str
        A registered pipeline-preset name (see :func:`list_pipeline_presets`).
    new_name : str
        The fresh name for the clone. Must be a non-empty string not already in
        the preset registry, and is also used to name each derived parameter row.
    allow_duplicate_params : bool, optional
        If True, allow a derived row whose content matches an existing row
        under a different name. Default False.
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
    # Before table imports, which connect to the database.
    if base_name not in _PIPELINE_PRESETS:
        raise ValueError(_unknown_pipeline_preset_message(base_name))
    if not isinstance(new_name, str) or not new_name.strip():
        raise ValueError(
            f"clone_pipeline_preset: new_name must be a non-empty string, got "
            f"{new_name!r}."
        )
    if new_name in _PIPELINE_PRESETS:
        raise ValueError(
            f"pipeline preset {new_name!r} is already registered. Choose a "
            f"fresh name; existing presets: {sorted(_PIPELINE_PRESETS)}."
        )
    if not overrides:
        raise ValueError(
            "clone_pipeline_preset requires at least one override (the knob to change); "
            "to register an alias of an existing preset under a new name use "
            "register_pipeline_preset()."
        )
    base = _PIPELINE_PRESETS[base_name]

    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import (
        PreprocessingWaveformParameters,
        SorterParameters,
    )

    # ``sorter`` is None for the single-key tables.
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

    # Routing needs every stage's blob to disambiguate an override.
    for stage in stages.values():
        rel = _require_preset_row(
            stage["table"],
            _base_restriction(stage),
            caller="clone_pipeline_preset",
            row=stage["base_row_name"],
            preset_name=base_name,
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

    touched, derived_params = _derive_preset_parameters(
        stages, overrides, base_name
    )

    def _fingerprint(stage, row):
        return parameter_row_fingerprint(
            stage["table_name"], row, sorter_keyed=stage["sorter"] is not None
        )

    # An existing ``new_name`` row is reused if identical, refused otherwise.
    with PreprocessingParameters._safe_context():
        for name in touched:
            stage = stages[name]
            derived_fp = _fingerprint(
                stage,
                {
                    "params": derived_params[name],
                    "params_schema_version": stage[
                        "base_params_schema_version"
                    ],
                    "job_kwargs": stage["base_job_kwargs"],
                    "sorter": stage["sorter"],
                    "execution_params": stage.get("base_execution_params"),
                    "execution_params_schema_version": stage.get(
                        "base_execution_params_schema_version"
                    ),
                },
            )

            new_restriction = {stage["pk_field"]: new_name}
            if stage["sorter"] is not None:
                new_restriction["sorter"] = stage["sorter"]
            existing = stage["table"] & new_restriction
            if existing:
                if _fingerprint(stage, existing.fetch1()) == derived_fp:
                    continue
                raise ValueError(
                    f"clone_pipeline_preset: a {stage['table_name']} row named "
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
        if "preprocessing" in touched:
            # Analyzer windows are keyed by preprocessing row name.
            display, metric = waveform_params_for_preprocessing(
                base.preprocessing_params_name
            )
            PreprocessingWaveformParameters.insert1(
                {
                    "preprocessing_params_name": new_name,
                    "display_waveform_params_name": display,
                    "metric_waveform_params_name": metric,
                },
                skip_duplicates=True,
            )

    derived_preset = base.model_copy(
        update={stages[name]["pk_field"]: new_name for name in touched}
    )
    register_pipeline_preset(new_name, derived_preset, validate_rows=True)
    return new_name
