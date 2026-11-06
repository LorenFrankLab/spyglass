"""Single source of truth for the shipped spike-sorting v2 recipes.

Owns the dated Frank-lab recipe row names, their parameter blobs, and the
pipeline presets. Lookup tables derive ``_DEFAULT_CONTENTS`` from the builders
here. Opens no DB connection at import, so schema modules import it without a
cycle.

``tests/spikesorting/v2/_recipe_constants.py`` keeps an independent copy of
these values; do not derive it from this module.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._params.analyzer_waveform import (
    AnalyzerWaveformParamsSchema,
)
from spyglass.spikesorting.v2._params.artifact_detection import (
    ArtifactDetectionParamsSchema,
)
from spyglass.spikesorting.v2._params.motion_estimation import (
    MotionEstimationParamsSchema,
)
from spyglass.spikesorting.v2._params.motion_interpolation import (
    MotionInterpolationParamsSchema,
)
from spyglass.spikesorting.v2._params.preprocessing import (
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2._params.sorter import (
    SorterExecutionParamsSchema,
    _get_sorter_schema,
)
from spyglass.spikesorting.v2._core.lookup_validation import _validate_params

# ---- Dated recipe row names ------------------------------------------------
# Sorters run ``filter=False``, so the region band lives on the preproc row.
HIPPOCAMPUS_PREPROC = "franklab_hippocampus_2026_06"
CORTEX_PREPROC = "franklab_cortex_2026_06"
NEUROPIXELS_PREPROC = "default_neuropixels"
# Artifact detection (the 500 uV schema default stays named "default").
ARTIFACT_100UV = "franklab_100uv_p07_2026_06"
ARTIFACT_50UV = "franklab_50uv_p07_2026_06"
# Sorter params (rate-keyed MS4 / MS5; KS4 neuropixels).
MS4_30KHZ = "franklab_30khz_ms4_2026_06"
MS4_20KHZ = "franklab_20khz_ms4_2026_06"
MS5_30KHZ = "franklab_30khz_ms5_2026_06"
KS4_NEUROPIXELS = "franklab_neuropixels_default"
# Containerized MS4: same science as the local 30 kHz row in a pinned
# Singularity image. The image tag is the MountainSort4 runtime version, not a
# SpikeInterface release; MS4_CONTAINER_SI_VERSION is the SI installed into it.
MS4_CONTAINER_IMAGE_TAG = "1.0.5"
MS4_CONTAINER_IMAGE = (
    f"spikeinterface/mountainsort4-base:{MS4_CONTAINER_IMAGE_TAG}"
)
MS4_CONTAINER_SI_VERSION = "0.104.3"
MS4_SINGULARITY_30KHZ = (
    "franklab_probe_hippocampus_30khz_ms4_singularity_2026_06"
)
# Analyzer waveforms: display unwhitened, metric whitened. Hippocampus uses
# 0.5/0.5 ms (narrower spikes), cortex 1.0/2.0 ms.
HIPPOCAMPUS_DISPLAY_WAVEFORMS = "franklab_hippocampus_actual_waveforms"
HIPPOCAMPUS_METRIC_WAVEFORMS = "franklab_hippocampus_metric_waveforms"
CORTEX_DISPLAY_WAVEFORMS = "franklab_cortex_actual_waveforms"
CORTEX_METRIC_WAVEFORMS = "franklab_cortex_metric_waveforms"
# Missing-metric behavior is recipe content, so changes ship as new dated rows.
FRANKLAB_CURATION_RULES = "franklab_default_auto_curation_2026_09"
V1_NOISE_CURATION_RULES = "v1_default_nn_noise_2026_09"


# ---- Per-stage default-row builders ---------------------------------------


def _params_schema_version(params: dict) -> int:
    """Return the authoritative schema version from a validated params blob."""
    return int(params["schema_version"])


def _lookup_row(name: str, params: dict) -> tuple:
    """Build a parameter-lookup row (no ``job_kwargs``)."""
    return (name, params, _params_schema_version(params), None)


def _sorter_row(
    sorter: str,
    name: str,
    params: dict,
    execution_params: dict | None = None,
) -> tuple:
    """Build a sorter-lookup row with versions copied from the blobs.

    Validates ``params`` against ``sorter``'s schema. Without
    ``execution_params`` the row gets the default local-execution blob.
    """
    params = _validate_params(_get_sorter_schema(sorter), params)
    exec_blob = (
        execution_params
        if execution_params is not None
        else SorterExecutionParamsSchema().model_dump()
    )
    return (
        sorter,
        name,
        params,
        _params_schema_version(params),
        None,
        exec_blob,
        _params_schema_version(exec_blob),
    )


def _waveform_row(name: str, params: dict) -> tuple:
    """Build an analyzer-waveform lookup row (no ``job_kwargs`` column)."""
    return (name, params, _params_schema_version(params))


def preprocessing_default_contents() -> tuple:
    """Return ``PreprocessingParameters._DEFAULT_CONTENTS``.

    Each row is ``(preprocessing_params_name, params_blob,
    params_schema_version, job_kwargs)``.
    """
    return (
        _lookup_row(
            "default",
            # Generic schema default, used by the clusterless preset.
            PreprocessingParamsSchema().model_dump(),
        ),
        _lookup_row(
            # 600 Hz high-pass for narrower hippocampal spikes; 1.5 ms
            # min-segment keeps short interval slivers.
            HIPPOCAMPUS_PREPROC,
            PreprocessingParamsSchema.model_validate(
                {
                    "bandpass_filter": {"freq_min": 600.0, "freq_max": 6000.0},
                    "min_segment_length": 0.0015,
                }
            ).model_dump(),
        ),
        _lookup_row(
            # Hippocampus recipe with a 300 Hz high-pass (wider waveforms).
            CORTEX_PREPROC,
            PreprocessingParamsSchema.model_validate(
                {
                    "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
                    "min_segment_length": 0.0015,
                }
            ).model_dump(),
        ),
        _lookup_row(
            NEUROPIXELS_PREPROC,
            # Phase shift is skipped without ``inter_sample_shift``.
            PreprocessingParamsSchema.model_validate(
                {
                    "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
                    "phase_shift": {"margin_ms": 100.0},
                }
            ).model_dump(),
        ),
        _lookup_row(
            "no_filter",
            PreprocessingParamsSchema.model_validate(
                {"bandpass_filter": None}
            ).model_dump(),
        ),
    )


def artifact_default_contents() -> tuple:
    """Return ``ArtifactDetectionParameters._DEFAULT_CONTENTS``.

    Each row is ``(artifact_detection_params_name, params_blob,
    params_schema_version, job_kwargs)``.
    """
    return (
        _lookup_row(
            "none",
            ArtifactDetectionParamsSchema(
                detect=False, amplitude_threshold_uv=None
            ).model_dump(),
        ),
        _lookup_row(
            "default",
            ArtifactDetectionParamsSchema().model_dump(),
        ),
        _lookup_row(
            ARTIFACT_100UV,
            ArtifactDetectionParamsSchema(
                amplitude_threshold_uv=100.0,
                proportion_above_threshold=0.7,
            ).model_dump(),
        ),
        _lookup_row(
            ARTIFACT_50UV,
            ArtifactDetectionParamsSchema(
                amplitude_threshold_uv=50.0,
                proportion_above_threshold=0.7,
            ).model_dump(),
        ),
    )


#: Cap (s) on unobserved time kept between continuity spans in motion
#: estimation. Each capped gap adds at most ``max_gap_s / bin_s`` bins.
MOTION_MAX_GAP_S = 30.0


def motion_estimation_default_contents() -> tuple:
    """Return ``MotionEstimationParameters._DEFAULT_CONTENTS``.

    Only the DREDge presets ship. None is validated for a probe.
    """
    return (
        _lookup_row(
            "dredge_v1",
            MotionEstimationParamsSchema(
                preset="dredge", max_gap_s=MOTION_MAX_GAP_S
            ).model_dump(),
        ),
        _lookup_row(
            "dredge_fast_v1",
            MotionEstimationParamsSchema(
                preset="dredge_fast", max_gap_s=MOTION_MAX_GAP_S
            ).model_dump(),
        ),
    )


#: ``kriging_force_extrapolate_v1`` matches SpikeInterface 0.104.3's DREDge
#: preset interpolation; ``kriging_remove_channels_v1`` swaps the border mode.
KRIGING_FORCE_EXTRAPOLATE = "kriging_force_extrapolate_v1"
KRIGING_REMOVE_CHANNELS = "kriging_remove_channels_v1"


def motion_interpolation_default_contents() -> tuple:
    """Return ``MotionInterpolationParameters._DEFAULT_CONTENTS``."""

    def _row(name: str, border_mode: str) -> tuple:
        params = MotionInterpolationParamsSchema(
            border_mode=border_mode,
            spatial_interpolation_method="kriging",
            sigma_um=20.0,
            p=2,
            num_closest=3,
        ).model_dump()
        return (name, params, _params_schema_version(params))

    return (
        _row(KRIGING_FORCE_EXTRAPOLATE, "force_extrapolate"),
        _row(KRIGING_REMOVE_CHANNELS, "remove_channels"),
    )


def motion_correction_default_contents() -> tuple:
    """Return ``MotionCorrectionParameters._DEFAULT_CONTENTS``.

    Each pairs a DREDge estimation row with its preset's interpolation.
    Neither is validated for a probe.
    """
    return (
        ("dredge_v1", "dredge_v1", KRIGING_FORCE_EXTRAPOLATE),
        ("dredge_fast_v1", "dredge_fast_v1", KRIGING_FORCE_EXTRAPOLATE),
    )


def _waveform_params(
    *, ms_before: float, ms_after: float, whiten: bool, purpose: str
) -> dict:
    """Validate + dump one analyzer-waveform params blob (20000 spikes)."""
    return AnalyzerWaveformParamsSchema(
        ms_before=ms_before,
        ms_after=ms_after,
        max_spikes_per_unit=20000,
        whiten=whiten,
        purpose=purpose,
    ).model_dump()


def waveform_params_default_contents() -> tuple:
    """Return ``AnalyzerWaveformParameters._DEFAULT_CONTENTS``."""
    return (
        _waveform_row(
            HIPPOCAMPUS_DISPLAY_WAVEFORMS,
            _waveform_params(
                ms_before=0.5, ms_after=0.5, whiten=False, purpose="display"
            ),
        ),
        _waveform_row(
            HIPPOCAMPUS_METRIC_WAVEFORMS,
            _waveform_params(
                ms_before=0.5, ms_after=0.5, whiten=True, purpose="metric"
            ),
        ),
        _waveform_row(
            CORTEX_DISPLAY_WAVEFORMS,
            _waveform_params(
                ms_before=1.0, ms_after=2.0, whiten=False, purpose="display"
            ),
        ),
        _waveform_row(
            CORTEX_METRIC_WAVEFORMS,
            _waveform_params(
                ms_before=1.0, ms_after=2.0, whiten=True, purpose="metric"
            ),
        ),
    )


# Shared by local and containerized MS4 rows. ``clip_size`` and
# ``detect_interval`` hold a ~1.33 ms window across rates.
_MS4_RATE_PARAMS: dict[int, dict] = {
    30000: {"adjacency_radius": 100.0},
    20000: {"adjacency_radius": 100.0, "clip_size": 27, "detect_interval": 7},
}


def _ms4_singularity_execution_params() -> dict:
    """Validated Singularity execution provenance for the containerized MS4 row.

    Pins the container's SpikeInterface version for reproducibility.
    """
    return SorterExecutionParamsSchema(
        backend="singularity",
        container_image=MS4_CONTAINER_IMAGE,
        installation_mode="pypi",
        spikeinterface_version=MS4_CONTAINER_SI_VERSION,
    ).model_dump()


def _ms4_singularity_sorter_row() -> tuple:
    """Build the containerized (Singularity, 30 kHz) MS4 ``SorterParameters`` row."""
    return _sorter_row(
        "mountainsort4",
        MS4_SINGULARITY_30KHZ,
        _MS4_RATE_PARAMS[30000],
        execution_params=_ms4_singularity_execution_params(),
    )


def sorter_default_contents() -> tuple:
    """Return ``SorterParameters._DEFAULT_CONTENTS``.

    Each row is ``(sorter, sorter_params_name, params_blob,
    params_schema_version, job_kwargs, execution_params_blob,
    execution_params_schema_version)``.
    """
    return (
        _sorter_row(
            "mountainsort4",
            MS4_30KHZ,
            _MS4_RATE_PARAMS[30000],
        ),
        _sorter_row(
            "mountainsort4",
            MS4_20KHZ,
            _MS4_RATE_PARAMS[20000],
        ),
        _sorter_row(
            # 30 kHz uses the schema-default snippet window.
            "mountainsort5",
            MS5_30KHZ,
            {},
        ),
        _sorter_row(
            # Matches the AIND KS4 capsule: stock KS4 except ``nblocks=5``.
            # No ``whiten`` key: KS4 whitens internally, so ``whiten=True``
            # would double-whiten.
            "kilosort4",
            KS4_NEUROPIXELS,
            {
                "nblocks": 5,
                "whitening_range": 32,
                "skip_kilosort_preprocessing": False,
                "highpass_cutoff": 300,
                "do_correction": True,
                "keep_good_only": False,
            },
        ),
        _sorter_row("spykingcircus2", "default", {}),
        _sorter_row("tridesclous2", "default", {}),
        _sorter_row(
            "clusterless_thresholder",
            "default",
            # The schema-default detect_threshold (100) in true uV.
            # ``noise_levels=[1.0]`` keeps it uV even if threshold_unit is lost.
            {"threshold_unit": "uv", "noise_levels": [1.0]},
        ),
        _ms4_singularity_sorter_row(),
    )


# ---- Pipeline presets ------------------------------------------------------
# probe_type is informational: the recipe is region + rate.
_REGION_PREPROC = {
    "hippocampus": HIPPOCAMPUS_PREPROC,
    "cortex": CORTEX_PREPROC,
}

# Region preprocessing recipe -> (display, metric) waveform-params rows.
_PREPROC_WAVEFORM_PARAMS = {
    HIPPOCAMPUS_PREPROC: (
        HIPPOCAMPUS_DISPLAY_WAVEFORMS,
        HIPPOCAMPUS_METRIC_WAVEFORMS,
    ),
    CORTEX_PREPROC: (CORTEX_DISPLAY_WAVEFORMS, CORTEX_METRIC_WAVEFORMS),
}


def waveform_params_for_preprocessing(
    preprocessing_params_name: str,
) -> tuple[str, str]:
    """Return the ``(display, metric)`` waveform-params row names for a recipe.

    Recipes other than hippocampus and cortex use the
    ``PreprocessingWaveformParameters`` row if one exists, else the wider
    cortex pair.

    Parameters
    ----------
    preprocessing_params_name : str
        The sort source's ``preprocessing_params_name``.

    Returns
    -------
    tuple of (str, str)
        The ``(display, metric)`` ``waveform_params_name`` pair.
    """
    if preprocessing_params_name in _PREPROC_WAVEFORM_PARAMS:
        return _PREPROC_WAVEFORM_PARAMS[preprocessing_params_name]
    from spyglass.spikesorting.v2.sorting import (
        PreprocessingWaveformParameters,
    )

    rows = (
        PreprocessingWaveformParameters
        & {"preprocessing_params_name": preprocessing_params_name}
    ).fetch("display_waveform_params_name", "metric_waveform_params_name")
    if len(rows[0]):
        return rows[0][0], rows[1][0]
    return CORTEX_DISPLAY_WAVEFORMS, CORTEX_METRIC_WAVEFORMS


# A new region must not silently inherit the cortex fallback.
assert set(_REGION_PREPROC.values()) <= set(_PREPROC_WAVEFORM_PARAMS), (
    "every _REGION_PREPROC recipe needs a _PREPROC_WAVEFORM_PARAMS mapping; "
    f"missing: {set(_REGION_PREPROC.values()) - set(_PREPROC_WAVEFORM_PARAMS)}"
)


_RATE_MS4_SORTER = {
    30000: MS4_30KHZ,
    20000: MS4_20KHZ,
}
_MS4_THRESHOLD_UNITS = "sigma of the whitened signal (~3)"
_MS4_NOTES = (
    "MountainSort4 detect_threshold is a multiple of the standard deviation "
    "of the ZCA-whitened signal (~3), not an absolute voltage and not a MAD "
    "multiplier. MS4 oversplits and does not track drift, so merge curation "
    "is expected (Kilosort is the Neuropixels-density alternative). Native "
    "MountainSort4 is included in the spikesorting-v2 extra and runs under "
    "numpy>=2 through v2's spikeextractors compatibility shim. Preflight "
    "imports mountainsort4 to check its runtime dependencies."
)


def _franklab_ms4_spec(probe_type: str, region: str, rate_hz: int) -> dict:
    """Build the ``_PipelinePreset`` field dict for an MS4 preset."""
    return dict(
        preprocessing_params_name=_REGION_PREPROC[region],
        artifact_detection_params_name=ARTIFACT_100UV,
        sorter="mountainsort4",
        sorter_params_name=_RATE_MS4_SORTER[rate_hz],
        metric_params_name="franklab_default",
        auto_curation_rules_name=FRANKLAB_CURATION_RULES,
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


def _franklab_ms4_singularity_spec() -> dict:
    """Build the ``_PipelinePreset`` field dict for the containerized MS4 preset.

    The probe-hippocampus-30 kHz MS4 spec pointed at the containerized sorter
    row. The backend lives on that row's ``execution_params``, not the preset.
    """
    spec = _franklab_ms4_spec("probe", "hippocampus", 30000)
    spec.update(
        sorter_params_name=MS4_SINGULARITY_30KHZ,
        intended_use=(
            "Frank-lab hippocampal polymer probes at 30 kHz, MountainSort4 run "
            "inside a pinned Singularity container -- a container execution "
            "option for the recommended-science MS4 recipe on modern hosts."
        ),
        notes=(
            "MountainSort4 detect_threshold is a multiple of the standard "
            "deviation of the ZCA-whitened signal (~3), not an absolute voltage "
            "and not a MAD multiplier. MS4 oversplits and does not track drift, "
            "so merge curation is expected. This containerized variant runs "
            f"MountainSort4 inside the pinned Singularity image "
            f"{MS4_CONTAINER_IMAGE} with container-side SpikeInterface pinned to "
            f"{MS4_CONTAINER_SI_VERSION} (installation_mode='pypi'), so the host "
            "can stay on the v2 numpy>=2 baseline. Native MS4 also supports "
            "numpy>=2. Preflight checks Singularity runtime "
            "availability and never silently falls back to local execution. "
            "run_v2_pipeline defaults to the native MS4 production recipe."
        ),
    )
    return spec


# MS5 alternatives; the tetrode and probe labels resolve to the same rows.
MS5_TETRODE_HIPPOCAMPUS_30KHZ = "franklab_tetrode_hippocampus_30khz_ms5_2026_06"
MS5_PROBE_HIPPOCAMPUS_30KHZ = "franklab_probe_hippocampus_30khz_ms5_2026_06"
# The run_v2_pipeline / preflight default.
DEFAULT_PIPELINE_PRESET = "franklab_probe_hippocampus_30khz_ms4_2026_06"

_MS5_NOTES = (
    "MountainSort5 detect_threshold is a multiple of the standard "
    "deviation of the whitened signal (~5.5, more conservative than "
    "MS4's 3) -- the same sigma scale, not a MAD multiplier. MS5 is an "
    "alternative to the default MountainSort4 hippocampus production recipe. "
    "Both run natively under numpy>=2. recommendation_status is 'alternative' "
    "(MS5 has no attested probe usage). The tetrode- and probe-labeled MS5 "
    "presets resolve to the same parameter rows (probe_type is "
    "informational)."
)


def _franklab_ms5_spec(probe_type: str) -> dict:
    """Build the ``_PipelinePreset`` field dict for a hippocampus-30 kHz MS5 preset."""
    if probe_type == "probe":
        intended_use = (
            "Frank-lab hippocampal probes at 30 kHz, MountainSort5 alternative. "
            "The default hippocampus recipe is MountainSort4: use the native "
            "franklab_probe_hippocampus_30khz_ms4_2026_06 preset in the standard "
            "v2 environment, or the containerized "
            f"{MS4_SINGULARITY_30KHZ} with Singularity/Apptainer."
        )
    else:
        intended_use = (
            f"Frank-lab hippocampal {probe_type}s at 30 kHz, MountainSort5 -- "
            "the same alternative recipe as the probe-labeled "
            f"{MS5_PROBE_HIPPOCAMPUS_30KHZ} under a tetrode label (probe_type "
            "is informational; both resolve to the same parameter rows)."
        )
    return dict(
        preprocessing_params_name=HIPPOCAMPUS_PREPROC,
        artifact_detection_params_name=ARTIFACT_100UV,
        sorter="mountainsort5",
        sorter_params_name=MS5_30KHZ,
        metric_params_name="franklab_default",
        auto_curation_rules_name=FRANKLAB_CURATION_RULES,
        probe_type=probe_type,
        target_region="hippocampus",
        sampling_rate_hz=30000,
        sorter_family="mountainsort5",
        recommendation_status="alternative",
        intended_use=intended_use,
        threshold_units="sigma of the whitened signal (~5.5)",
        notes=_MS5_NOTES,
    )


def pipeline_preset_specs() -> dict[str, dict]:
    """Return ``{preset_name: _PipelinePreset field dict}`` for every preset.

    Plain dicts so the catalog does not import the preset model.
    """
    return {
        "franklab_tetrode_hippocampus_30khz_ms4_2026_06": _franklab_ms4_spec(
            "tetrode", "hippocampus", 30000
        ),
        "franklab_probe_hippocampus_30khz_ms4_2026_06": _franklab_ms4_spec(
            "probe", "hippocampus", 30000
        ),
        "franklab_probe_cortex_30khz_ms4_2026_06": _franklab_ms4_spec(
            "probe", "cortex", 30000
        ),
        "franklab_probe_hippocampus_20khz_ms4_2026_06": _franklab_ms4_spec(
            "probe", "hippocampus", 20000
        ),
        "franklab_probe_cortex_20khz_ms4_2026_06": _franklab_ms4_spec(
            "probe", "cortex", 20000
        ),
        MS5_TETRODE_HIPPOCAMPUS_30KHZ: _franklab_ms5_spec("tetrode"),
        MS5_PROBE_HIPPOCAMPUS_30KHZ: _franklab_ms5_spec("probe"),
        "franklab_clusterless_2026_06": dict(
            preprocessing_params_name="default",
            artifact_detection_params_name="default",
            sorter="clusterless_thresholder",
            sorter_params_name="default",
            # No clustered units to curate.
            metric_params_name="minimal",
            auto_curation_rules_name="none",
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
                "threshold, not a MAD multiplier. Preprocessing and artifact "
                "detection use the 'default' rows."
            ),
        ),
        "franklab_neuropixels_ks4_2026_06": dict(
            preprocessing_params_name=NEUROPIXELS_PREPROC,
            artifact_detection_params_name="none",
            sorter="kilosort4",
            sorter_params_name=KS4_NEUROPIXELS,
            metric_params_name="neuropixels_default",
            auto_curation_rules_name=V1_NOISE_CURATION_RULES,
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
                "KS4. Artifact detection is 'none': this preset applies no "
                "Spyglass artifact mask. Internal preprocessing and drift correction "
                "do not substitute for artifact rejection."
            ),
        ),
        MS4_SINGULARITY_30KHZ: _franklab_ms4_singularity_spec(),
        "franklab_concat_hippocampus_30khz_ms5_2026_09": {
            **_franklab_ms5_spec("probe"),
            "recommendation_status": "experimental",
            "intended_use": (
                "Frank-lab same-day chronic hippocampal probes at 30 kHz: "
                "concatenate a SessionGroup's members into one "
                "ConcatenatedRecording and sort once (MountainSort5). Run it "
                "with run_v2_concat_pipeline."
            ),
            "notes": (
                "Detect and mask artifacts per member before concatenation. "
                "Otherwise the same MS5 "
                "hippocampus recipe as the single-session MS5 alternative; MS5 runs "
                "under the v2 numpy>=2 baseline. Run it with "
                "run_v2_concat_pipeline. Marked experimental: validate the "
                "recipe on your own same-day data before relying on it."
            ),
        },
    }


def quality_metric_default_rows() -> list[dict]:
    """Shipped ``QualityMetricParameters`` rows, as raw insert input.

    ``franklab_default`` and ``neuropixels_default`` carry the full metric set
    (including the ``nn_advanced`` PCA metric); ``minimal`` computes only
    ``snr``, ``isi_violation`` and ``firing_rate``.
    """
    # nn_advanced is a PCA metric, so the full rows set skip_pc_metrics=False.
    nn_kwargs = {
        "n_components": 7,
        "n_neighbors": 5,
        "max_spikes": 20000,
        "min_spikes": 10,
        "seed": 0,
    }
    isi_kwargs = {"isi_threshold_ms": 2.0, "min_isi_ms": 0.0}
    full_metrics = [
        "snr",
        "isi_violation",
        "firing_rate",
        "num_spikes",
        "presence_ratio",
        "amplitude_cutoff",
        "nn_advanced",
    ]
    full_metric_kwargs = {
        "snr": {"peak_sign": "neg"},
        "isi_violation": isi_kwargs,
        "nn_advanced": nn_kwargs,
    }
    rows = [
        {
            "metric_params_name": name,
            "metric_names": full_metrics,
            "metric_kwargs": full_metric_kwargs,
            "skip_pc_metrics": False,
        }
        for name in ("franklab_default", "neuropixels_default")
    ]
    rows.append(
        {
            "metric_params_name": "minimal",
            "metric_names": ["snr", "isi_violation", "firing_rate"],
            "metric_kwargs": {
                "snr": {"peak_sign": "neg"},
                "isi_violation": isi_kwargs,
            },
            "skip_pc_metrics": True,
        }
    )
    return rows


def auto_curation_default_payloads() -> list[tuple[dict, list[dict]]]:
    """Shipped ``AutoCurationRules`` masters with their ordered rule rows."""
    return [
        (
            {
                "auto_curation_rules_name": "none",
                "auto_merge_preset": "none",
            },
            [],
        ),
        (
            {
                "auto_curation_rules_name": V1_NOISE_CURATION_RULES,
                "auto_merge_preset": "none",
            },
            [
                {
                    "rule_index": 0,
                    "rule_name": "nn_noise",
                    "metric_name": "nn_noise_overlap",
                    "operator": ">",
                    "threshold": 0.1,
                    "label": "noise",
                    "missing_policy": "pass",
                },
                {
                    "rule_index": 1,
                    "rule_name": "nn_reject",
                    "metric_name": "nn_noise_overlap",
                    "operator": ">",
                    "threshold": 0.1,
                    "label": "reject",
                    "missing_policy": "pass",
                },
            ],
        ),
        (
            {
                "auto_curation_rules_name": "similarity_merge",
                "auto_merge_preset": "similarity_correlograms",
            },
            [],
        ),
        (
            # ISI violators are ``reject`` (not ``mua``) so every unit
            # selection policy denies them. Merges stay manual. Needs a metric
            # row computing nn_advanced and isi_violation.
            {
                "auto_curation_rules_name": FRANKLAB_CURATION_RULES,
                "auto_merge_preset": "none",
            },
            [
                {
                    "rule_index": 0,
                    "rule_name": "nn_noise",
                    "metric_name": "nn_noise_overlap",
                    "operator": ">",
                    "threshold": 0.1,
                    "label": "noise",
                    "missing_policy": "pass",
                },
                {
                    "rule_index": 1,
                    "rule_name": "isi_reject",
                    "metric_name": "isi_violation",
                    "operator": ">",
                    "threshold": 0.02,
                    "label": "reject",
                    "missing_policy": "pass",
                },
            ],
        ),
    ]
