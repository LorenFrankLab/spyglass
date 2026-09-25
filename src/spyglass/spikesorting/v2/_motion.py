"""SpikeInterface motion estimation: parameters, serialization, summaries.

``motion_to_storage_dict`` / ``motion_from_storage_dict`` round-trip a
``Motion`` through a DataJoint-blob-safe dict; ``motion_max_abs_displacement_um``
and ``motion_n_temporal_bins`` summarize one for a stored row.

``resolve_estimation_params`` turns a ``MotionEstimationParameters`` blob into
the fully resolved configuration the estimator passes to SpikeInterface, and
``resolved_params_hash`` content-addresses it.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection; SpikeInterface is imported lazily inside the functions that need
it.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json

import numpy as np

#: Version of the motion-estimation algorithm this module implements (the
#: SpikeInterface call sequence, peak filter, noise estimate and resolution
#: rules). Part of every estimate's identity; bump it when any of those change
#: in a way that can change a stored estimate.
MOTION_ALGORITHM_VERSION = 1

#: Peak waveform window (ms) used to localize peaks. SpikeInterface 0.104.3's
#: ``compute_motion`` extracts exactly this window in its detect-and-localize
#: pipeline (``preprocessing/motion.py:387``), which the estimator reproduces.
#: A peak is kept only when this whole window lies inside one statistics span.
LOCALIZATION_MS_BEFORE = 0.1
LOCALIZATION_MS_AFTER = 0.3

#: SpikeInterface methods the estimator supports for each step. Every shipped
#: preset uses exactly these; others would need their own resolution rules
#: (for example, a torch detection device) before they could be pinned.
_DETECT_METHODS = ("locally_exclusive",)
_LOCALIZE_METHODS = (
    "center_of_mass",
    "monopolar_triangulation",
    "grid_convolution",
)
_ESTIMATE_METHODS = ("dredge_ap",)

#: Signature parameters the estimator binds itself, per step: never persisted
#: as configuration.
_DETECT_BOUND = frozenset(
    {"self", "recording", "noise_levels", "return_output"}
)
_LOCALIZE_BOUND = frozenset({"self", "recording", "parents", "return_output"})
_ESTIMATE_BOUND = frozenset(
    {
        "recording",
        "peaks",
        "peak_locations",
        "extra_outputs",
        "progress_bar",
        "verbose",
        "margin_um",
        "precomputed_D_C_maxdisp",
    }
)

#: Callables a resolved configuration may name, by their canonical string.
#: ``dredge_ap``'s ``post_transform`` defaults to ``numpy.log1p``.
_NAMED_CALLABLES = {"numpy.log1p": np.log1p}

#: Random-chunk budget of the noise estimate: SpikeInterface's
#: ``get_random_recording_slices`` defaults (``core/recording_tools.py:461-468``),
#: which the span sampler (``_sorting_dispatch._sample_statistics_spans``) also
#: uses.
_NOISE_METHOD = "mad"
_NOISE_NUM_CHUNKS = 20
_NOISE_CHUNK_DURATION = "500ms"


def motion_to_storage_dict(motion) -> dict:
    """Flatten a SpikeInterface ``Motion`` to a DataJoint-blob-safe dict.

    The stored blob keeps every field the ``Motion`` constructor needs so
    :func:`motion_from_storage_dict` reconstructs the object exactly:

    - ``displacement``: list (one entry per recording segment) of 2-D
      ``float`` arrays, each shape ``(n_temporal_bins, n_spatial_bins)`` (um).
    - ``temporal_bins_s``: list (per segment) of 1-D bin-center arrays (s).
    - ``spatial_bins_um``: a single 1-D array of window centers (um),
      ``shape == (n_spatial_bins,)`` -- shared across segments.
    - ``direction``: the motion axis (``"x"`` / ``"y"`` / ``"z"``).
    - ``interpolation_method``: how displacement interpolates between bins.

    Arrays stay as NumPy arrays (DataJoint's blob codec round-trips them);
    only the container shape is normalized so the rehydrate side is
    deterministic regardless of how the codec returns nested lists.
    """
    return {
        "displacement": [np.asarray(d) for d in motion.displacement],
        "temporal_bins_s": [np.asarray(t) for t in motion.temporal_bins_s],
        "spatial_bins_um": np.asarray(motion.spatial_bins_um),
        "direction": str(motion.direction),
        "interpolation_method": str(motion.interpolation_method),
    }


def motion_from_storage_dict(blob: dict):
    """Rebuild a SpikeInterface ``Motion`` from a stored blob.

    Inverse of :func:`motion_to_storage_dict`. Coerces each entry back to a
    clean NumPy array before constructing ``Motion`` so a blob codec that
    returns the per-segment lists as object arrays (rather than Python lists)
    still rehydrates to the 2-D / 1-D shapes ``Motion`` asserts on. Every key
    is read directly (no defaulting): :func:`motion_to_storage_dict` always
    writes all five, so a missing key means a corrupt blob and should raise
    rather than silently rehydrate a wrong motion axis.
    """
    from spikeinterface.core.motion import Motion

    return Motion(
        [np.asarray(d) for d in blob["displacement"]],
        [np.asarray(t) for t in blob["temporal_bins_s"]],
        np.asarray(blob["spatial_bins_um"]),
        direction=str(blob["direction"]),
        interpolation_method=str(blob["interpolation_method"]),
    )


def motion_max_abs_displacement_um(motion) -> float:
    """Largest absolute displacement (um) over all segments / bins / windows.

    The single-number drift-severity summary used to flag high-drift
    sessions. NaNs are NOT masked -- a non-finite estimate surfaces in the
    stored metric rather than being silently hidden by a ``nan``-aware max.
    """
    flat = np.concatenate([np.asarray(d).ravel() for d in motion.displacement])
    return float(np.max(np.abs(flat)))


def motion_n_temporal_bins(motion) -> int:
    """Total number of temporal bins across all recording segments."""
    return int(sum(np.asarray(t).shape[0] for t in motion.temporal_bins_s))


def _signature_defaults(func, bound: frozenset) -> dict:
    """Return ``func``'s defaulted parameters, minus the ``bound`` ones."""
    return {
        name: param.default
        for name, param in inspect.signature(func).parameters.items()
        if name not in bound
        and param.default is not inspect.Parameter.empty
        and param.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }


def _signature_names(func, bound: frozenset) -> set[str]:
    """Return ``func``'s named parameters, minus the ``bound`` ones."""
    return {
        name
        for name, param in inspect.signature(func).parameters.items()
        if name not in bound
        and param.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }


def _canonical(value, reference=None):
    """Encode ``value`` in the JSON-stable form a resolved config stores.

    NumPy scalars and arrays become Python numbers and lists, tuples become
    lists, and a callable becomes its registered name. An integer given where
    the reference value (the preset's or the signature default) is a float is
    stored as a float, so ``8`` and ``8.0`` resolve to one configuration.
    """
    if callable(value) and not isinstance(value, type):
        for name, func in _NAMED_CALLABLES.items():
            if value is func:
                return name
        raise TypeError(
            f"cannot store callable {value!r} in a motion configuration; "
            f"only {sorted(_NAMED_CALLABLES)} are supported."
        )
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and isinstance(reference, float)
    ):
        return float(value)
    return value


def _resolve_step(
    step: str, merged: dict, preset_step: dict, defaults: dict, allowed: set
) -> dict:
    """Layer a step's merged preset/user kwargs over its signature defaults.

    A value's numeric type is canonicalized against the preset's value for
    that key when the preset names it, else against the signature default.

    Raises
    ------
    ValueError
        If ``merged`` names a key that is not a parameter of the method.
    """
    unknown = sorted(set(merged) - allowed - {"method"})
    if unknown:
        raise ValueError(
            f"{step} names {unknown}, which are not parameters of method "
            f"{merged['method']!r}; allowed: {sorted(allowed)}."
        )
    resolved = {k: _canonical(v) for k, v in defaults.items()}
    for key, value in merged.items():
        reference = preset_step.get(key, resolved.get(key))
        resolved[key] = _canonical(value, reference=reference)
    return resolved


def resolve_estimation_params(params: dict) -> dict:
    """Return the fully resolved motion-estimation configuration.

    SpikeInterface merges a user's per-step overrides one level deep over the
    preset step, ``dict(preset_step, **user_step)``
    (``preprocessing/motion.py:260-268``, ``_update_motion_kwargs``), and
    leaves every argument the preset does not name at the called function's
    own default. ``get_motion_parameters_preset`` does not report all of those
    defaults: ``dredge_ap``'s live only in its own signature
    (``sortingcomponents/motion/dredge.py:139-175``; the registration class's
    ``run`` has none, ``dredge.py:92-108``). This function applies the same
    merge and then writes every default out explicitly, so the stored
    configuration is exactly what the estimator passes and no SpikeInterface
    default stays implicit:

    - ``detect_kwargs``: the detection class's ``__init__`` defaults, then the
      merged preset/user step. ``noise_levels`` is supplied at run time.
    - ``localize_peaks_kwargs``: the localization class's ``__init__``
      defaults, then the merged step.
    - ``estimate_motion_kwargs``: ``dredge_ap``'s signature defaults, then
      ``estimate_motion``'s (which win: ``estimate_motion`` passes
      ``direction``/``rigid``/``win_*`` to the method explicitly,
      ``motion_estimation.py:108-122``), then the merged step, with
      ``device`` pinned to ``"cpu"`` (``None`` would pick CUDA when available,
      ``dredge.py:984-985``).
    - ``localization_window_ms``: the fixed peak waveform window.
    - ``noise_levels_kwargs``: the noise estimator and its seeded chunk budget.

    Values are canonical (plain Python numbers, lists, ``None``; ``post_transform``
    as ``"numpy.log1p"``), so :func:`resolved_params_hash` is stable.

    Parameters
    ----------
    params : dict
        A ``MotionEstimationParameters.params`` blob (validated or as fetched).

    Returns
    -------
    dict
        The resolved configuration.

    Raises
    ------
    ValueError
        If a step selects an unsupported method or names a key that is not a
        parameter of the selected method.
    """
    from spikeinterface.preprocessing.motion import motion_options_preset
    from spikeinterface.sortingcomponents.motion.dredge import dredge_ap
    from spikeinterface.sortingcomponents.motion.motion_estimation import (
        estimate_motion,
    )
    from spikeinterface.sortingcomponents.peak_detection import (
        detect_peak_methods,
    )
    from spikeinterface.sortingcomponents.peak_localization import (
        peak_localization_methods,
    )

    from spyglass.spikesorting.v2._lookup_validation import _jsonable_blob
    from spyglass.spikesorting.v2._params.motion_estimation import (
        MotionEstimationParamsSchema,
    )

    validated = MotionEstimationParamsSchema.model_validate(
        _jsonable_blob(params)
    )
    preset = motion_options_preset[validated.preset]
    merged = {
        step: dict(preset[step], **getattr(validated, step))
        for step in (
            "detect_kwargs",
            "localize_peaks_kwargs",
            "estimate_motion_kwargs",
        )
    }

    def _method(step: str, allowed: tuple) -> str:
        method = merged[step].get("method")
        if method not in allowed:
            raise ValueError(
                f"{step}['method'] must be one of {list(allowed)}; got "
                f"{method!r}."
            )
        return method

    detect_cls = detect_peak_methods[_method("detect_kwargs", _DETECT_METHODS)]
    detect = _resolve_step(
        "detect_kwargs",
        merged["detect_kwargs"],
        preset["detect_kwargs"],
        _signature_defaults(detect_cls.__init__, _DETECT_BOUND),
        _signature_names(detect_cls.__init__, _DETECT_BOUND),
    )

    localize_cls = peak_localization_methods[
        _method("localize_peaks_kwargs", _LOCALIZE_METHODS)
    ]
    localize = _resolve_step(
        "localize_peaks_kwargs",
        merged["localize_peaks_kwargs"],
        preset["localize_peaks_kwargs"],
        _signature_defaults(localize_cls.__init__, _LOCALIZE_BOUND),
        _signature_names(localize_cls.__init__, _LOCALIZE_BOUND),
    )

    _method("estimate_motion_kwargs", _ESTIMATE_METHODS)
    estimate_explicit = _signature_names(estimate_motion, _ESTIMATE_BOUND)
    estimate_defaults = {
        **{
            k: v
            for k, v in _signature_defaults(dredge_ap, _ESTIMATE_BOUND).items()
            if k not in estimate_explicit
        },
        **_signature_defaults(estimate_motion, _ESTIMATE_BOUND),
    }
    estimate = _resolve_step(
        "estimate_motion_kwargs",
        merged["estimate_motion_kwargs"],
        preset["estimate_motion_kwargs"],
        estimate_defaults,
        estimate_explicit | _signature_names(dredge_ap, _ESTIMATE_BOUND),
    )
    estimate["device"] = "cpu"

    return {
        "preset": validated.preset,
        "detect_kwargs": detect,
        "select_kwargs": {},
        "localize_peaks_kwargs": localize,
        "estimate_motion_kwargs": estimate,
        "localization_window_ms": {
            "ms_before": LOCALIZATION_MS_BEFORE,
            "ms_after": LOCALIZATION_MS_AFTER,
        },
        "noise_levels_kwargs": {
            "method": _NOISE_METHOD,
            "num_chunks_per_segment": _NOISE_NUM_CHUNKS,
            "chunk_duration": _NOISE_CHUNK_DURATION,
            "seed": int(validated.noise_levels_seed),
        },
    }


def resolved_params_hash(resolved: dict) -> str:
    """SHA-256 of a resolved configuration's canonical JSON encoding."""
    encoded = json.dumps(
        _canonical(resolved), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def spikeinterface_step_kwargs(resolved: dict) -> tuple[dict, dict, dict]:
    """Decode a resolved configuration into SpikeInterface call kwargs.

    Returns fresh copies (``dredge_ap`` mutates a passed ``xcorr_kw`` dict,
    ``dredge.py:231-232``) with named callables restored.

    Parameters
    ----------
    resolved : dict
        Output of :func:`resolve_estimation_params` (or its stored blob).

    Returns
    -------
    tuple of dict
        ``(detect_kwargs, localize_peaks_kwargs, estimate_motion_kwargs)``,
        each including its ``method`` key as ``compute_motion`` expects.
    """
    detect = copy.deepcopy(dict(resolved["detect_kwargs"]))
    localize = copy.deepcopy(dict(resolved["localize_peaks_kwargs"]))
    estimate = copy.deepcopy(dict(resolved["estimate_motion_kwargs"]))
    estimate["post_transform"] = _NAMED_CALLABLES[estimate["post_transform"]]
    return detect, localize, estimate
