"""Single source of truth for the shipped spike-sorting v2 recipes.

Owns the dated Frank-lab recipe row names, their parameter blobs, and the
pipeline-preset bundles + UX metadata. The per-stage parameter Lookup
tables derive their ``_DEFAULT_CONTENTS`` from the builders here
(:func:`preprocessing_default_contents` / :func:`artifact_default_contents`
/ :func:`sorter_default_contents`), and ``_pipeline_presets`` builds
``_PIPELINE_PRESETS`` from :func:`pipeline_preset_specs` -- so a recipe is
defined in exactly one place.

No DB connection or ``dj.schema`` activation at import: it builds the
blobs from the ``_params`` Pydantic schemas only, so it imports without a
database connection and the schema modules import it without a cycle.
("No DB" means no connection/activation -- NOT "no DataJoint installed":
like every spyglass module it still transitively pulls DataJoint /
SpikeInterface via the framework's package ``__init__``.)

``tests/spikesorting/v2/_recipe_constants.py`` keeps an INDEPENDENT
hand-written copy of these recipe values that the parity tests compare
against -- do not derive those literals from this module, or the parity
check becomes a tautology.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._params.analyzer_waveform import (
    AnalyzerWaveformParamsSchema,
)
from spyglass.spikesorting.v2._params.artifact_detection import (
    ArtifactDetectionParamsSchema,
)
from spyglass.spikesorting.v2._params.preprocessing import (
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2._params.sorter import (
    SorterExecutionParamsSchema,
    _get_sorter_schema,
)
from spyglass.spikesorting.v2.utils import _validate_params

# ---- Dated recipe row names (single source for the cross-references) -------
# Preprocessing: region high-pass (filtering happens at the preproc stage; the
# sorter runs ``filter=False``, so the region band lives on the preproc row).
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
# Containerized MS4 (probe / polymer, 30 kHz): a first-class, reproducible
# execution path for modern hosts. MS4's algorithm backend (ml_ms4alg) is a
# numpy<2-era package that does not install under the v2 numpy>=2 baseline, so
# these rows run the SAME scientific params as ``MS4_30KHZ`` inside a pinned
# container -- the host stays on numpy>=2 while the MS4 runtime lives in the
# image. The image is SpikeInterface's published mountainsort4 base image pinned
# to a release tag (the same image reference works for Docker and -- pulled as a
# SIF -- for Singularity/Apptainer), and the container-side SpikeInterface is
# pinned to the same release (installation_mode="pypi") so the row is
# reproducible by content. A local MS4 row and these containerized rows are
# DISTINCT named rows (different sorter_params_name); the container backend is
# tracked provenance, not a runtime override. Singularity is the Frank-lab HPC
# target; Docker is shipped for workstation hosts.
MS4_CONTAINER_SI_VERSION = "0.104.3"
MS4_CONTAINER_IMAGE = (
    f"spikeinterface/mountainsort4-base:{MS4_CONTAINER_SI_VERSION}"
)
MS4_SINGULARITY_30KHZ = (
    "franklab_probe_hippocampus_30khz_ms4_singularity_2026_06"
)
MS4_DOCKER_30KHZ = "franklab_probe_hippocampus_30khz_ms4_docker_2026_06"
MS4_SINGULARITY_20KHZ = (
    "franklab_probe_hippocampus_20khz_ms4_singularity_2026_06"
)
MS4_DOCKER_20KHZ = "franklab_probe_hippocampus_20khz_ms4_docker_2026_06"
# Analyzer waveform recipes (region-specific window; display=unwhitened,
# metric=whitened). Hippocampal spikes are denser/tighter, so they take a
# narrower 0.5/0.5 ms window; cortical waveforms are broader, so they take the
# wider 1.0/2.0 ms window. Both subsample 20000 spikes.
HIPPOCAMPUS_DISPLAY_WAVEFORMS = "franklab_hippocampus_actual_waveforms"
HIPPOCAMPUS_METRIC_WAVEFORMS = "franklab_hippocampus_metric_waveforms"
CORTEX_DISPLAY_WAVEFORMS = "franklab_cortex_actual_waveforms"
CORTEX_METRIC_WAVEFORMS = "franklab_cortex_metric_waveforms"


# ---- Per-stage default-row builders ---------------------------------------


def _params_schema_version(params: dict) -> int:
    """Return the authoritative schema version from a validated params blob."""
    return int(params["schema_version"])


def _lookup_row(name: str, params: dict, job_kwargs=None) -> tuple:
    """Build a parameter-lookup row with the version copied from ``params``."""
    return (name, params, _params_schema_version(params), job_kwargs)


def _sorter_row(
    sorter: str,
    name: str,
    params: dict,
    job_kwargs=None,
    execution_params: dict | None = None,
) -> tuple:
    """Build a sorter-lookup row with versions copied from the blobs.

    Each row is ``(sorter, sorter_params_name, params_blob,
    params_schema_version, job_kwargs, execution_params_blob,
    execution_params_schema_version)`` -- matching the ``SorterParameters``
    heading order. A row that omits ``execution_params`` gets the schema-default
    local-execution blob.
    """
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
        job_kwargs,
        exec_blob,
        _params_schema_version(exec_blob),
    )


def _waveform_row(name: str, params: dict) -> tuple:
    """Build an analyzer-waveform lookup row (no ``job_kwargs`` column).

    Each row is ``(waveform_params_name, params_blob, params_schema_version)``;
    the version is copied from the validated blob.
    """
    return (name, params, _params_schema_version(params))


def preprocessing_default_contents() -> tuple:
    """Return ``PreprocessingParameters._DEFAULT_CONTENTS``.

    Each row is ``(preprocessing_params_name, params_blob,
    params_schema_version, job_kwargs)``.
    """
    return (
        _lookup_row(
            "default",
            # v2's schema-default preproc (300 Hz / 6000 Hz bandpass, median
            # reference, 1.0 s min-segment). Not a production recipe -- the
            # franklab production presets use the dated region rows below; this
            # is the generic default the clusterless preset and ad-hoc callers
            # use. ``whiten`` defaults to None (whitening is deferred to the
            # sorter), so the default-constructed schema needs no override.
            PreprocessingParamsSchema().model_dump(),
        ),
        _lookup_row(
            # Production hippocampus recipe (June 2026): 600 Hz high-pass
            # (hippocampal spikes are denser/narrower than cortical ones),
            # 6000 Hz low-pass, 1.5 ms min-segment (production keeps the
            # short interval slivers the 1.0 s shipped default drops). Filtering
            # happens at this preproc stage; the MS4 sorter runs ``filter=False``
            # (see _params/sorter.py), so the region high-pass lives on the
            # preproc row, never the sorter row.
            HIPPOCAMPUS_PREPROC,
            PreprocessingParamsSchema.model_validate(
                {
                    "bandpass_filter": {"freq_min": 600.0, "freq_max": 6000.0},
                    "min_segment_length": 0.0015,
                }
            ).model_dump(),
        ),
        _lookup_row(
            # Production cortex recipe (June 2026): identical to the
            # hippocampus recipe with a 300 Hz high-pass (cortical waveforms
            # are wider).
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
            # Blessed Neuropixels recipe: bandpass + ADC phase-shift. The
            # phase-shift is a safe no-op until the recording carries an
            # ``inter_sample_shift`` property (the runtime logs a skip and
            # never fails), so selecting this preset never breaks a
            # non-multiplexed recording.
            PreprocessingParamsSchema.model_validate(
                {
                    "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
                    "phase_shift": {"margin_ms": 100.0},
                }
            ).model_dump(),
        ),
        _lookup_row(
            "no_filter",
            # ``bandpass_filter=None`` is a real disable: the runtime
            # skips the filter step entirely, so "no_filter" applies no
            # bandpass at all.
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
            # Production artifact recipe (June 2026): 100 uV amplitude
            # threshold, 0.7 proportion-above-threshold, 1.0 ms removal window
            # -- far more aggressive than the 500 uV shipped "default".
            ARTIFACT_100UV,
            ArtifactDetectionParamsSchema(
                amplitude_threshold_uv=100.0,
                proportion_above_threshold=0.7,
            ).model_dump(),
        ),
        _lookup_row(
            # More aggressive 50 uV production variant (same proportion-above
            # and removal window).
            ARTIFACT_50UV,
            ArtifactDetectionParamsSchema(
                amplitude_threshold_uv=50.0,
                proportion_above_threshold=0.7,
            ).model_dump(),
        ),
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
    """Return ``AnalyzerWaveformParameters._DEFAULT_CONTENTS``.

    Region-specific display (unwhitened) and metric (whitened) recipes:
    hippocampus 0.5/0.5 ms (denser/tighter spikes), cortex 1.0/2.0 ms (broader
    waveforms); both subsample 20000 spikes. Each row is
    ``(waveform_params_name, params_blob, params_schema_version)``.
    """
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


# Rate-keyed MS4 scientific params, shared by the local rate-keyed rows AND the
# containerized rows so a containerized row is byte-identical science to its
# local sibling -- the container backend is the only difference. The sorter runs
# ``filter=False`` (the preproc stage already bandpassed), so the only
# rate-dependent knobs are ``clip_size`` / ``detect_interval`` (they hold the
# same ~1.33 ms physical window across rates); ``adjacency_radius`` is set
# explicitly to 100 um (also the schema default).
_MS4_RATE_PARAMS: dict[int, dict] = {
    30000: {"adjacency_radius": 100.0},
    20000: {"adjacency_radius": 100.0, "clip_size": 27, "detect_interval": 7},
}

# Containerized MS4 rows: the probe/polymer hippocampus rate × container-backend
# matrix. Each name pairs with the rate's ``_MS4_RATE_PARAMS`` and the backend's
# execution provenance. Singularity is the Frank-lab HPC target; Docker is for
# workstation hosts.
_MS4_CONTAINER_SORTER_NAMES: dict[tuple[str, int], str] = {
    ("singularity", 30000): MS4_SINGULARITY_30KHZ,
    ("docker", 30000): MS4_DOCKER_30KHZ,
    ("singularity", 20000): MS4_SINGULARITY_20KHZ,
    ("docker", 20000): MS4_DOCKER_20KHZ,
}


def _ms4_container_execution_params(backend: str) -> dict:
    """Validated container execution provenance for a containerized MS4 row.

    The given container ``backend`` (``"docker"`` or ``"singularity"``) with the
    pinned image, and container-side SpikeInterface pinned to an explicit version
    (``installation_mode="pypi"``) so the runtime install is reproducible by row
    content -- not floating via ``installation_mode="auto"``.
    """
    return SorterExecutionParamsSchema(
        backend=backend,
        container_image=MS4_CONTAINER_IMAGE,
        installation_mode="pypi",
        spikeinterface_version=MS4_CONTAINER_SI_VERSION,
    ).model_dump()


def _ms4_container_sorter_rows() -> tuple:
    """Build the containerized MS4 ``SorterParameters`` rows (rate × backend).

    Each row's scientific params are IDENTICAL to the local rate-keyed row of
    the same rate (shared ``_MS4_RATE_PARAMS`` source); the only difference is the
    tracked container execution backend. The distinct ``sorter_params_name`` (and
    the duplicate-content guard folding ``execution_params`` in) lets each coexist
    with its local sibling instead of forking provenance.
    """
    return tuple(
        _sorter_row(
            "mountainsort4",
            name,
            _validate_params(
                _get_sorter_schema("mountainsort4"), _MS4_RATE_PARAMS[rate]
            ),
            execution_params=_ms4_container_execution_params(backend),
        )
        for (backend, rate), name in _MS4_CONTAINER_SORTER_NAMES.items()
    )


def sorter_default_contents() -> tuple:
    """Return ``SorterParameters._DEFAULT_CONTENTS``.

    Each row is ``(sorter, sorter_params_name, params_blob,
    params_schema_version, job_kwargs, execution_params_blob,
    execution_params_schema_version)``. The local sorter rows (MS4 rate-keyed,
    MS5, KS4, SC2/TDC2, clusterless) plus the containerized MS4 rows
    (:func:`_ms4_container_sorter_rows`).
    """
    return (
        _sorter_row(
            # Production MountainSort4 recipe (June 2026), 30 kHz. Rate-keyed:
            # the sorter runs ``filter=False`` (the preproc stage already
            # bandpassed), so the sorter band is inert and the only
            # rate-dependent knobs are ``clip_size`` / ``detect_interval``. The
            # tetrode/probe/region distinction lives on the preproc row + the
            # preset, not here. ``adjacency_radius`` is set explicitly to 100
            # um (also the schema default).
            "mountainsort4",
            MS4_30KHZ,
            _validate_params(
                _get_sorter_schema("mountainsort4"), _MS4_RATE_PARAMS[30000]
            ),
        ),
        _sorter_row(
            # 20 kHz variant: ``clip_size=27`` / ``detect_interval=7`` hold the
            # same ~1.33 ms physical window at the lower rate; everything else
            # matches the 30 kHz row.
            "mountainsort4",
            MS4_20KHZ,
            _validate_params(
                _get_sorter_schema("mountainsort4"), _MS4_RATE_PARAMS[20000]
            ),
        ),
        _sorter_row(
            # MS5 is region-agnostic (filter=False, like MS4), so the row is
            # rate-keyed; 30 kHz uses the schema-default snippet window.
            "mountainsort5",
            MS5_30KHZ,
            _validate_params(_get_sorter_schema("mountainsort5"), {}),
        ),
        _sorter_row(
            # Kilosort4 Neuropixels recipe matched to the AIND
            # aind-ephys-spikesort-kilosort4 capsule params.json (and
            # int-brain-lab/ibl-sorter): the only scientifically-meaningful
            # deviation from stock KS4 is non-rigid drift correction
            # (``nblocks=5`` vs the stock ``1``); the rest pins KS4's stock
            # whitening/preprocessing config explicitly. KS4 does its own
            # high-pass + common-reference + ZCA whitening internally
            # (``skip_kilosort_preprocessing=False``, ``whitening_range=32``).
            # There is deliberately NO ``whiten`` key: KS4 has no such param,
            # and the v2 runtime only runs its external float64 whitening when
            # the sorter params carry ``whiten=True`` (see
            # ``_sorting_dispatch.py``), so omitting it keeps the signal
            # whitened exactly once (by KS4) -- adding ``whiten=True`` here
            # would double-whiten.
            "kilosort4",
            KS4_NEUROPIXELS,
            _validate_params(
                _get_sorter_schema("kilosort4"),
                {
                    "nblocks": 5,
                    "whitening_range": 32,
                    "skip_kilosort_preprocessing": False,
                    "highpass_cutoff": 300,
                    "do_correction": True,
                    "keep_good_only": False,
                },
            ),
        ),
        _sorter_row(
            "spykingcircus2",
            "default",
            _validate_params(_get_sorter_schema("spykingcircus2"), {}),
        ),
        _sorter_row(
            "tridesclous2",
            "default",
            _validate_params(_get_sorter_schema("tridesclous2"), {}),
        ),
        _sorter_row(
            "clusterless_thresholder",
            "default",
            # ClusterlessThresholderSchema is currently schema_version=4:
            # v2 dropped ``outputs`` / ``random_chunk_kwargs``;
            # v3 made ``noise_levels`` optional (None -> SI MAD);
            # v4 added ``threshold_unit``.
            _validate_params(
                _get_sorter_schema("clusterless_thresholder"),
                # ``threshold_unit="uv"`` makes the shipped
                # ``detect_threshold=100`` a TRUE 100 microvolt threshold:
                # the runtime scales the recording to uV (via the stored
                # NWB gain) before detection, rather than treating it as a
                # MAD multiplier. (For Frank-lab data gain==1 uV/count so
                # 100 "uv" == 100 counts == 100 uV either way.) The
                # explicit ``noise_levels=[1.0]`` is the equivalent
                # advanced override and is kept as a belt-and-suspenders
                # regression guard against the 1,400x noise_levels
                # divergence; the runtime uses it verbatim (explicit
                # noise_levels take precedence over ``threshold_unit``).
                # The smoke / synthetic-fixture rows set
                # ``threshold_unit="mad"`` EXPLICITLY (no noise_levels) so SI
                # computes per-channel MAD and the threshold tracks the
                # recording's noise floor -- they do not rely on the 'uv'
                # default unit.
                #
                # THIS shipped ``default`` row sets ``threshold_unit="uv"``
                # explicitly and takes detect_threshold from the schema
                # default (100); the runtime ``scale_to_uV`` makes that a
                # true 100 uV threshold.
                {"threshold_unit": "uv", "noise_levels": [1.0]},
            ),
        ),
    ) + _ms4_container_sorter_rows()


# ---- Pipeline presets ------------------------------------------------------
# Production region preproc + rate-keyed MS4 sorter rows the franklab MS4
# presets bundle. probe_type is informational (the recipe is region + rate),
# so the tetrode- and probe-hippocampus-30 kHz presets resolve identically.
_REGION_PREPROC = {
    "hippocampus": HIPPOCAMPUS_PREPROC,
    "cortex": CORTEX_PREPROC,
}

# A sort's analyzer waveform window is region-specific, resolved from the SAME
# signal that sets the region filter cutoff: the source preprocessing recipe.
# This maps each region preprocessing recipe to its ``(display, metric)``
# waveform-params row pair. Lives here (not on ``_PipelinePreset``, which is
# ``extra="forbid"``) so no preset-schema change is needed.
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

    Only hippocampus and cortex have region-tuned analyzer windows; every other
    preprocessing recipe -- custom, multi-region, AND the shipped non-tetrode
    recipes (e.g. ``default``, ``default_neuropixels``, ``no_filter``) -- falls
    back to the wider cortex pair (1.0/2.0 ms). This is deliberate: the analyzer
    window is tuned for hippocampal vs cortical tetrode waveforms, and adding a
    tuned window for another region is a tracked-row-and-mapping change, not a
    silent default. The fallback never mixes windows within a sort.

    Parameters
    ----------
    preprocessing_params_name : str
        The sort source's ``preprocessing_params_name``.

    Returns
    -------
    tuple of (str, str)
        The ``(display, metric)`` ``waveform_params_name`` pair.
    """
    return _PREPROC_WAVEFORM_PARAMS.get(
        preprocessing_params_name,
        (CORTEX_DISPLAY_WAVEFORMS, CORTEX_METRIC_WAVEFORMS),
    )


# Drift guard: every region in ``_REGION_PREPROC`` must have a waveform-params
# mapping, so a future region added to the preset map cannot silently inherit
# the cortex fallback. (Non-region shipped recipes -- neuropixels / no_filter --
# intentionally fall back; see ``waveform_params_for_preprocessing``.)
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
    "is expected (Kilosort is the Neuropixels-density alternative). Runtime "
    "note: MS4's algorithm backend (ml_ms4alg) is a numpy<2-era package that "
    "does not install under the v2 numpy>=2 baseline, so these presets need a "
    "numpy<2 environment; the shipped run_v2_pipeline default is the "
    "MountainSort5 recipe, which runs as-is (preflight reports this via the "
    "sorter_runtime_available check)."
)


def _franklab_ms4_spec(probe_type: str, region: str, rate_hz: int) -> dict:
    """Build the ``_PipelinePreset`` field dict for an MS4 preset."""
    return dict(
        preprocessing_params_name=_REGION_PREPROC[region],
        artifact_detection_params_name=ARTIFACT_100UV,
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


def _franklab_ms4_container_spec(backend: str, rate_hz: int) -> dict:
    """Build the ``_PipelinePreset`` field dict for a containerized MS4 preset.

    Reuses the probe-hippocampus MS4 spec at ``rate_hz`` (same preproc /
    artifact / scientific sorter params), then overrides the sorter row to the
    containerized one for the given container ``backend`` and stamps the
    execution-backend discovery metadata + a notes/intended-use explaining the
    modern-host (numpy>=2) container path. ``recommendation_status`` stays
    ``"production"``: these are the recommended-science MS4 paths on modern
    hosts. ``run_v2_pipeline``'s default remains MountainSort5.
    """
    rate_khz = rate_hz // 1000
    spec = _franklab_ms4_spec("probe", "hippocampus", rate_hz)
    spec.update(
        sorter_params_name=_MS4_CONTAINER_SORTER_NAMES[(backend, rate_hz)],
        execution_backend=backend,
        container_image=MS4_CONTAINER_IMAGE,
        intended_use=(
            f"Frank-lab hippocampal polymer probes at {rate_khz} kHz, "
            f"MountainSort4 run inside a pinned {backend.capitalize()} container "
            "-- the recommended-science MS4 path on modern hosts where MS4's "
            "ml_ms4alg backend cannot run locally under the v2 numpy>=2 baseline."
        ),
        notes=(
            "MountainSort4 detect_threshold is a multiple of the standard "
            "deviation of the ZCA-whitened signal (~3), not an absolute voltage "
            "and not a MAD multiplier. MS4 oversplits and does not track drift, "
            "so merge curation is expected. This containerized variant runs MS4's "
            f"ml_ms4alg backend inside the pinned {backend.capitalize()} image "
            f"{MS4_CONTAINER_IMAGE} with container-side SpikeInterface pinned to "
            f"{MS4_CONTAINER_SI_VERSION} (installation_mode='pypi'), so the host "
            "can stay on the v2 numpy>=2 baseline -- the local MS4 presets need "
            f"numpy<2, this one does not. Preflight checks {backend.capitalize()} "
            "runtime availability and never silently falls back to local "
            "execution. run_v2_pipeline's default remains MountainSort5."
        ),
    )
    return spec


def pipeline_preset_specs() -> dict[str, dict]:
    """Return ``{preset_name: _PipelinePreset field dict}`` for every preset.

    ``_pipeline_presets`` constructs the validated ``_PipelinePreset`` objects
    from these specs; keeping them as plain dicts here lets the catalog stay
    free of any dependency on the preset model (no import cycle).
    """
    specs = {
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
        "franklab_tetrode_hippocampus_30khz_ms5_2026_06": dict(
            preprocessing_params_name=HIPPOCAMPUS_PREPROC,
            artifact_detection_params_name=ARTIFACT_100UV,
            sorter="mountainsort5",
            sorter_params_name=MS5_30KHZ,
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
        "franklab_clusterless_2026_06": dict(
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
        "franklab_neuropixels_ks4_2026_06": dict(
            preprocessing_params_name=NEUROPIXELS_PREPROC,
            artifact_detection_params_name="none",
            sorter="kilosort4",
            sorter_params_name=KS4_NEUROPIXELS,
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
    # Containerized MS4 presets: probe-hippocampus rate × container-backend
    # matrix, each bundling the matching containerized sorter row.
    specs.update(
        {
            name: _franklab_ms4_container_spec(backend, rate)
            for (backend, rate), name in _MS4_CONTAINER_SORTER_NAMES.items()
        }
    )
    return specs


def _assert_preset_execution_matches_sorter_rows() -> None:
    """Drift guard: a preset's execution metadata matches its sorter row.

    ``describe_pipeline_presets`` surfaces the ``execution_backend`` /
    ``container_image`` discovery metadata DB-free, but the authoritative
    execution provenance lives on the ``SorterParameters.execution_params`` blob
    of the row each preset references. Both are built from this catalog, so
    assert at import that they agree -- a preset cannot advertise a backend/image
    that its sorter row does not actually carry.
    """
    sorter_execution = {
        (row[0], row[1]): row[5]  # (sorter, name) -> execution_params blob
        for row in sorter_default_contents()
    }
    for name, spec in pipeline_preset_specs().items():
        exec_blob = sorter_execution.get(
            (spec["sorter"], spec["sorter_params_name"])
        )
        if exec_blob is None:
            # A preset referencing a non-default sorter row; the
            # presets-reference-shipped-rows test covers that separately.
            continue
        backend = spec.get("execution_backend", "local")
        image = spec.get("container_image", "") or ""
        assert exec_blob["backend"] == backend, (
            f"preset {name!r} advertises execution_backend={backend!r} but its "
            f"sorter row {spec['sorter_params_name']!r} carries "
            f"backend={exec_blob['backend']!r}"
        )
        assert (exec_blob["container_image"] or "") == image, (
            f"preset {name!r} advertises container_image={image!r} but its "
            f"sorter row {spec['sorter_params_name']!r} carries "
            f"container_image={exec_blob['container_image']!r}"
        )


_assert_preset_execution_matches_sorter_rows()
