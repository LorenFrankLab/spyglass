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
    intended_use: str = ""  # one-line "when to reach for this pipeline preset"
    threshold_units: str = ""  # detection-threshold units (sigma / µV)
    notes: str = ""  # key assumptions (probe geometry, sampling rate, etc.)


# Production region preproc + rate-keyed MS4 sorter rows the franklab MS4
# presets bundle. probe_type is informational (the recipe is region + rate),
# so the tetrode- and probe-hippocampus-30 kHz presets resolve identically.
_REGION_PREPROC = {
    "hippocampus": "franklab_hippocampus_2026_06",
    "cortex": "franklab_cortex_2026_06",
}
_RATE_MS4_SORTER = {
    30000: "franklab_30khz_ms4_2026_06",
    20000: "franklab_20khz_ms4_2026_06",
}
_MS4_THRESHOLD_UNITS = "sigma of the whitened signal (~3)"
_MS4_NOTES = (
    "MountainSort4 detect_threshold is a multiple of the standard deviation "
    "of the ZCA-whitened signal (~3), not an absolute voltage and not a MAD "
    "multiplier. MS4 oversplits and does not track drift, so merge curation "
    "is expected (Kilosort is the Neuropixels-density alternative). Runtime "
    "note: MS4's algorithm backend (ml_ms4alg) is a numpy<2-era package that "
    "does not install under the v2 numpy>=2 baseline, so these presets need a "
    "numpy<2 environment; the shipped run_v2_pipeline default is the "
    "MountainSort5 recipe, which runs as-is (preflight reports this via the "
    "sorter_runtime_available check)."
)


def _franklab_ms4_preset(
    probe_type: str, region: str, rate_hz: int
) -> "_PipelinePreset":
    """Build a production MS4 preset for a (probe_type, region, rate)."""
    return _PipelinePreset(
        preprocessing_params_name=_REGION_PREPROC[region],
        artifact_detection_params_name="franklab_100uv_p07_2026_06",
        sorter="mountainsort4",
        sorter_params_name=_RATE_MS4_SORTER[rate_hz],
        probe_type=probe_type,
        target_region=region,
        sampling_rate_hz=rate_hz,
        sorter_family="mountainsort4",
        adjacency_radius_um=100.0,
        recommendation_status="production",
        intended_use=(
            f"Frank-lab {region} {probe_type}s at {rate_hz // 1000} kHz "
            "(MountainSort4 production recipe)."
        ),
        threshold_units=_MS4_THRESHOLD_UNITS,
        notes=_MS4_NOTES,
    )


_PIPELINE_PRESETS: dict[str, _PipelinePreset] = {
    "franklab_tetrode_hippocampus_30khz_ms4_2026_06": _franklab_ms4_preset(
        "tetrode", "hippocampus", 30000
    ),
    "franklab_probe_hippocampus_30khz_ms4_2026_06": _franklab_ms4_preset(
        "probe", "hippocampus", 30000
    ),
    "franklab_probe_cortex_30khz_ms4_2026_06": _franklab_ms4_preset(
        "probe", "cortex", 30000
    ),
    "franklab_probe_hippocampus_20khz_ms4_2026_06": _franklab_ms4_preset(
        "probe", "hippocampus", 20000
    ),
    "franklab_probe_cortex_20khz_ms4_2026_06": _franklab_ms4_preset(
        "probe", "cortex", 20000
    ),
    "franklab_tetrode_hippocampus_30khz_ms5_2026_06": _PipelinePreset(
        preprocessing_params_name="franklab_hippocampus_2026_06",
        artifact_detection_params_name="franklab_100uv_p07_2026_06",
        sorter="mountainsort5",
        sorter_params_name="franklab_30khz_ms5_2026_06",
        probe_type="tetrode",
        target_region="hippocampus",
        sampling_rate_hz=30000,
        sorter_family="mountainsort5",
        recommendation_status="alternative",
        intended_use=(
            "Frank-lab hippocampal tetrodes at 30 kHz, MountainSort5 -- the "
            "shipped run_v2_pipeline default because it runs under the v2 "
            "numpy>=2 baseline (the MS4 production recipe needs numpy<2)."
        ),
        threshold_units="sigma of the whitened signal (~5.5)",
        notes=(
            "MountainSort5 detect_threshold is a multiple of the standard "
            "deviation of the whitened signal (~5.5, more conservative than "
            "MS4's 3) -- the same sigma scale, not a MAD multiplier. MS5 is the "
            "shipped run_v2_pipeline default because it runs under numpy>=2; "
            "MS4 is the Frank-lab production recipe but its ml_ms4alg backend "
            "needs numpy<2. recommendation_status stays 'alternative' (MS5 has "
            "no attested probe usage); the function default is a separate, "
            "runnability-driven choice."
        ),
    ),
    "franklab_clusterless_2026_06": _PipelinePreset(
        preprocessing_params_name="default",
        artifact_detection_params_name="default",
        sorter="clusterless_thresholder",
        sorter_params_name="default",
        target_region="hippocampus",
        sorter_family="clusterless_thresholder",
        recommendation_status="production",
        intended_use=(
            "Peak detection only (no clustering); feeds the clusterless "
            "decoding pipeline."
        ),
        threshold_units="µV (100 µV)",
        notes=(
            "The 'default' clusterless SorterParameters row sets "
            "threshold_unit='uv' with detect_threshold=100, so traces are "
            "scaled to microvolts before detection -- a true 100 µV "
            "threshold, not a MAD multiplier. Preproc/artifact rows are "
            "unchanged from the prior clusterless preset."
        ),
    ),
    "franklab_neuropixels_ks4_2026_06": _PipelinePreset(
        preprocessing_params_name="default_neuropixels",
        artifact_detection_params_name="none",
        sorter="kilosort4",
        sorter_params_name="franklab_neuropixels_default",
        probe_type="neuropixels",
        sampling_rate_hz=30000,
        sorter_family="kilosort4",
        recommendation_status="experimental",
        intended_use=(
            "Neuropixels (30 kHz) with Kilosort4, matched to the AIND "
            "aind-ephys-spikesort-kilosort4 recipe. Experimental -- "
            "community-grounded, not Frank-lab-attested; KS4 is "
            "non-deterministic and typically needs a GPU."
        ),
        threshold_units=(
            "KS4 template-matching projection thresholds "
            "(Th_universal=9 / Th_learned=8), not µV or a noise multiple"
        ),
        notes=(
            "Sorter params match the AIND aind-ephys-spikesort-kilosort4 "
            "params.json (and agree with int-brain-lab/ibl-sorter): the only "
            "scientifically-meaningful deviation from stock KS4 is non-rigid "
            "drift correction (nblocks=5 vs stock 1). KS4 does its own "
            "high-pass + common-reference + ZCA whitening internally "
            "(skip_kilosort_preprocessing=False, whitening_range=32); the row "
            "carries no 'whiten' key, so v2's external float64 whitening stays "
            "off and the signal is whitened EXACTLY ONCE (by KS4). The preproc "
            "row applies the ADC phase-shift KS4 cannot, plus a bandpass; "
            "because KS4 also common-references (do_CAR=true), set the sort "
            "group's reference_mode='none' to avoid double-referencing before "
            "KS4. Artifact detection is 'none' (KS4's internal preprocessing "
            "and drift handling stand in for amplitude masking)."
        ),
    ),
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
        ``"preprocessing"`` / ``"artifact_detection"`` / ``"sorter"``),
        ``params_row_name``, ``key`` (dotted path into the validated blob), and
        ``value``. Parameter-row entries also carry ``params_schema_version`` and
        ``job_kwargs`` so the display shows the full row resolved by the preset,
        not just the core params blob. Preset entries include the human-facing
        ``threshold_units`` field so detection thresholds are never unitless.
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
    return pd.DataFrame(rows, columns=_PRESET_DETAIL_COLUMNS)


# Alias: shorter discovery name for the same helper.
describe_preset = describe_pipeline_preset
