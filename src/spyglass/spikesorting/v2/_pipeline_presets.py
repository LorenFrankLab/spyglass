"""Pipeline presets: the dated franklab production recipe bundles.

Extracted verbatim from ``pipeline.py`` (behavior-preserving) to keep the
orchestration facade small. ``pipeline.py`` re-exports the public names
(``list_pipeline_presets``, ``describe_pipeline_presets``,
``describe_pipeline_preset``, ``describe_preset``) so notebook import paths
are unchanged.
"""

from __future__ import annotations

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
    artifact_detection_params_name: str
    sorter: str
    sorter_params_name: str
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
    artifact-detection, and sorter parameter rows -- so you can see exactly what
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
        ``"preprocessing"`` / ``"artifact_detection"`` / ``"sorter"`` /
        ``"sorter_execution"``), ``params_row_name``, ``key`` (dotted path into
        the validated blob), and ``value``. Parameter-row entries also carry
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
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
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
            ("threshold_units", preset.threshold_units),
            ("intended_use", preset.intended_use),
            ("notes", preset.notes),
        )
    ]
    for stage, row_name, row in (
        (
            "preprocessing",
            preset.preprocessing_params_name,
            _fetch_params(
                PreprocessingParameters,
                {"preprocessing_params_name": preset.preprocessing_params_name},
                "PreprocessingParameters",
            ),
        ),
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
        ),
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
        ),
    ):
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
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import SorterParameters

    checks = (
        (
            PreprocessingParameters,
            {"preprocessing_params_name": preset.preprocessing_params_name},
            "PreprocessingParameters",
            preset.preprocessing_params_name,
        ),
        (
            ArtifactDetectionParameters,
            {
                "artifact_detection_params_name": (
                    preset.artifact_detection_params_name
                )
            },
            "ArtifactDetectionParameters",
            preset.artifact_detection_params_name,
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
