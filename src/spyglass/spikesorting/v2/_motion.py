"""SpikeInterface motion estimation: parameters, serialization, summaries.

``motion_to_storage_dict`` / ``motion_from_storage_dict`` round-trip a
``Motion`` through a DataJoint-blob-safe dict; ``motion_max_abs_displacement_um``
and ``motion_n_temporal_bins`` summarize one for a stored row.

``resolve_estimation_params`` turns a ``MotionEstimationParameters`` blob into
the fully resolved configuration the estimator passes to SpikeInterface, and
``resolved_params_hash`` content-addresses it.

``estimate_motion_in_spans`` estimates motion from the valid samples of one
recording: it reproduces SpikeInterface's ``compute_motion`` with explicit
noise levels from the statistics spans and keeps only peaks whose localization
window lies inside one statistics span.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection; SpikeInterface is imported lazily inside the functions that need
it.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from typing import NamedTuple

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


class MotionDiagnostics(NamedTuple):
    """Compact evidence summary of one motion estimate.

    Attributes
    ----------
    n_peaks_detected : int
        Peaks the detector found on the masked recording.
    n_peaks_kept : int
        Peaks whose localization window lies inside one statistics span; only
        these reach the estimator.
    peaks_per_temporal_bin : numpy.ndarray
        ``(n_temporal_bins,)`` int64 count of kept peaks in each of the
        estimate's temporal bins.
    noise_levels : numpy.ndarray
        ``(n_channels,)`` float64 per-channel noise (recording units) the
        detection threshold was scaled by.
    """

    n_peaks_detected: int
    n_peaks_kept: int
    peaks_per_temporal_bin: np.ndarray
    noise_levels: np.ndarray


def normalize_spans(spans) -> list[tuple[int, int]]:
    """Return ``spans`` as a list of ``(start, end)`` Python-int tuples."""
    return [
        (int(a), int(b))
        for a, b in np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    ]


def _check_estimation_spans(
    n_samples: int,
    continuity_spans: list[tuple[int, int]],
    statistics_spans: list[tuple[int, int]],
) -> None:
    """Validate the span inputs of :func:`estimate_motion_in_spans`.

    Raises
    ------
    ValueError
        If the input has more than one continuity span (not supported yet),
        the continuity span does not cover the recording, or the statistics
        spans are empty, unsorted, overlapping or outside the recording.
    """
    if len(continuity_spans) != 1:
        raise ValueError(
            f"Motion estimation: the input has {len(continuity_spans)} "
            "continuity spans (acquisition gaps or member joins); "
            "discontinuous inputs are not supported yet. Estimate each "
            "gap-free recording separately."
        )
    if continuity_spans != [(0, int(n_samples))]:
        raise ValueError(
            f"Motion estimation: continuity span {continuity_spans[0]} does "
            f"not cover the recording's {n_samples} samples."
        )
    if not statistics_spans:
        raise ValueError("Motion estimation: no statistics spans were given.")
    previous_end = 0
    for start, end in statistics_spans:
        if not previous_end <= start < end <= n_samples:
            raise ValueError(
                "Motion estimation: statistics spans must be sorted, "
                f"disjoint, non-empty and inside [0, {n_samples}); got "
                f"{statistics_spans}."
            )
        previous_end = end


def peaks_within_spans(
    sample_index, spans: list[tuple[int, int]], *, n_before: int, n_after: int
) -> np.ndarray:
    """Mark peaks whose waveform window lies inside a single span.

    A peak at frame ``s`` reads frames ``[s - n_before, s + n_after)``
    (SpikeInterface's ``ExtractDenseWaveforms``,
    ``core/node_pipeline.py:363-364``). It is kept only when some span
    ``[a, b)`` holds all of them, so no kept localization reads a masked
    sample, the zero padding past the recording's ends, or across a span edge.

    Parameters
    ----------
    sample_index : numpy.ndarray
        ``(n_peaks,)`` peak frames.
    spans : list[tuple[int, int]]
        Sorted, disjoint, half-open frame spans.
    n_before, n_after : int
        Keyword-only. Waveform frames before and from the peak.

    Returns
    -------
    numpy.ndarray
        ``(n_peaks,)`` bool mask of kept peaks.
    """
    sample_index = np.asarray(sample_index, dtype=np.int64)
    starts = np.array([a for a, _ in spans], dtype=np.int64)
    ends = np.array([b for _, b in spans], dtype=np.int64)
    span = np.searchsorted(starts, sample_index, side="right") - 1
    found = span >= 0
    span = np.clip(span, 0, len(spans) - 1)
    return (
        found
        & (sample_index - n_before >= starts[span])
        & (sample_index + n_after <= ends[span])
    )


def _shank_labels(recording) -> set:
    """Distinct shanks among a recording's channels.

    A shank is an ``(electrode group, probe_shank)`` pair, read from the
    electrode-table properties a v2 recording artifact carries; a recording
    without them (a synthetic one) falls back to its probe's shank ids, and a
    recording with neither counts as one shank.
    """
    keys = recording.get_property_keys()
    n_channels = recording.get_num_channels()
    if "probe_shank" in keys:
        groups = (
            recording.get_property("group")
            if "group" in keys
            else [""] * n_channels
        )
        shanks = recording.get_property("probe_shank")
        return {(str(g), str(s)) for g, s in zip(groups, shanks)}
    if recording.get_property("contact_vector") is not None:
        shank_ids = recording.get_probe().shank_ids
        if shank_ids is not None:
            return {str(s) for s in shank_ids}
    return {""}


def check_estimation_eligibility(recording, resolved_params: dict) -> None:
    """Refuse a recording whose geometry cannot support the recipe.

    Flattens a reloaded artifact's constant third coordinate in place
    (:func:`._recording_geometry.flatten_planar_geometry`), then requires:

    - finite, distinct 2D contact positions;
    - one shank (nonrigid windows are built along the motion axis only, so
      several shanks would be registered as one column);
    - a probe SpikeInterface can attach (``get_probe`` builds a dummy probe
      from the locations; the estimator's spatial bins come from it,
      ``sortingcomponents/motion/motion_utils.py:160-170``);
    - a depth extent along the motion axis of at least the detection
      ``radius_um``;
    - for a nonrigid recipe, room for at least one step of spatial windows.
      SpikeInterface computes ``(extent + 2 * margin) // win_step_um`` with
      ``margin = win_margin_um`` or ``-win_scale_um / 2``
      (``motion_utils.py:72-78``) and, when that is below 1, only warns and
      builds a window layout other than the configured one
      (``motion_utils.py:80-88``); this raises instead.

    Parameters
    ----------
    recording : si.BaseRecording
        The recording to estimate on. Mutated only by the flattening.
    resolved_params : dict
        Output of :func:`resolve_estimation_params`.

    Raises
    ------
    ValueError
        Naming the first failed condition.
    """
    import warnings

    from spyglass.spikesorting.v2._recording_geometry import (
        flatten_planar_geometry,
    )

    if (
        recording.get_property("contact_vector") is None
        and recording.get_property("location") is None
    ):
        raise ValueError(
            "Motion estimation: the recording carries no contact positions, "
            "so no probe can be attached."
        )
    flatten_planar_geometry(recording)
    positions = np.asarray(recording.get_channel_locations(), dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            "Motion estimation: contact positions must be 2D after planar "
            f"flattening; got shape {positions.shape}."
        )
    if not np.isfinite(positions).all():
        raise ValueError(
            "Motion estimation: contact positions must be finite; got "
            f"{positions.tolist()}."
        )
    if len(np.unique(np.round(positions, 6), axis=0)) != len(positions):
        raise ValueError(
            "Motion estimation: two or more contacts share a position "
            f"({positions.tolist()}); motion cannot be estimated on "
            "coincident contacts."
        )
    shanks = _shank_labels(recording)
    if len(shanks) > 1:
        raise ValueError(
            f"Motion estimation: the recording spans {len(shanks)} shanks "
            f"({sorted(shanks)}); estimate one shank at a time."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            recording.get_probe()
    except Exception as exc:  # probeinterface raises assorted types
        raise ValueError(
            "Motion estimation: no probe can be attached to the recording "
            f"after planar flattening ({exc})."
        ) from exc

    estimate = resolved_params["estimate_motion_kwargs"]
    axis = "xyz".index(estimate["direction"])
    if axis >= positions.shape[1]:
        raise ValueError(
            f"Motion estimation: motion direction {estimate['direction']!r} "
            "is not an axis of the 2D contact positions."
        )
    extent = float(np.ptp(positions[:, axis]))
    radius = float(resolved_params["detect_kwargs"]["radius_um"])
    if extent < radius:
        raise ValueError(
            f"Motion estimation: the contacts span {extent:g} um along the "
            f"motion axis, less than the detection radius_um ({radius:g}); "
            "the probe is too short to track motion with this recipe."
        )
    if not estimate["rigid"]:
        margin = estimate["win_margin_um"]
        if margin is None:
            margin = -float(estimate["win_scale_um"]) / 2.0
        n_steps = (extent + 2.0 * float(margin)) // float(
            estimate["win_step_um"]
        )
        if n_steps < 1:
            raise ValueError(
                f"Motion estimation: the contacts span {extent:g} um along "
                "the motion axis, too short for the nonrigid windows "
                f"(win_step_um={estimate['win_step_um']:g}, "
                f"win_scale_um={estimate['win_scale_um']:g}, "
                f"win_margin_um={estimate['win_margin_um']}); SpikeInterface "
                "would silently change the window layout. Use a rigid recipe."
            )


def estimation_noise_levels(
    recording, statistics_spans: list[tuple[int, int]], noise_kwargs: dict
) -> np.ndarray:
    """Per-channel noise the detection threshold is scaled by.

    With spans that exclude samples, the MAD of seeded random samples drawn
    only inside the statistics spans (``_sorting_dispatch.
    cache_span_noise_levels``, the estimate the analyzer noise levels use),
    so masked zeros and joins do not lower the threshold. With one span
    covering the recording, SpikeInterface's own ``get_noise_levels`` -- the
    call ``compute_motion`` makes (``preprocessing/motion.py:359``) -- with
    its random chunks seeded, since the unseeded call is not repeatable
    (``core/recording_tools.py:461-468``). ``n_jobs=1`` keeps the chunk
    average in a fixed order.

    Parameters
    ----------
    recording : si.BaseRecording
        The masked recording. Its ``noise_level_mad_raw`` property is set.
    statistics_spans : list[tuple[int, int]]
        Half-open frame spans of valid samples.
    noise_kwargs : dict
        ``resolved_params["noise_levels_kwargs"]``.

    Returns
    -------
    numpy.ndarray
        ``(n_channels,)`` float64 noise levels in recording units.
    """
    from spikeinterface.core import get_noise_levels

    from spyglass.spikesorting.v2._sorting_dispatch import (
        cache_span_noise_levels,
    )

    seed = int(noise_kwargs["seed"])
    levels = cache_span_noise_levels(
        recording,
        statistics_spans,
        return_in_uV=False,
        seed=seed,
        method=noise_kwargs["method"],
    )
    if levels is None:
        levels = get_noise_levels(
            recording,
            return_in_uV=False,
            method=noise_kwargs["method"],
            force_recompute=True,
            random_slices_kwargs={
                "method": "full_random",
                "num_chunks_per_segment": int(
                    noise_kwargs["num_chunks_per_segment"]
                ),
                "chunk_duration": noise_kwargs["chunk_duration"],
                "seed": seed,
            },
            n_jobs=1,
        )
    return np.asarray(levels, dtype=np.float64)


def use_estimation_clock(recording) -> None:
    """Put a one-span recording on its estimation clock, in place.

    Within a continuity span, estimation time advances at exactly ``1 / fs``
    from the span's first timestamp. A recording without a time vector is
    already on that clock. A recording with one (every v2 artifact read back
    from NWB) has it replaced by ``t0 + i / fs``: by the continuity-span
    definition no step between its consecutive timestamps exceeds ``1.5 / fs``
    (``boundary_spans_from_timestamps``). Besides defining the
    clock, this keeps ``dredge_ap``'s peak-time lookup working: it maps all
    peak frames with one fancy index (``sortingcomponents/motion/dredge.py:
    227``), which an HDF5-backed time vector refuses for repeated or
    unordered frames.

    Parameters
    ----------
    recording : si.BaseRecording
        Single-segment recording; its time information is replaced.
    """
    if not recording.has_time_vector():
        return
    t0 = float(recording.sample_index_to_time(0))
    recording.reset_times()
    recording.shift_times(t0)


def estimate_motion_in_spans(
    recording,
    *,
    statistics_spans,
    continuity_spans,
    resolved_params: dict,
    job_kwargs: dict | None = None,
):
    """Estimate motion from the valid samples of one continuous recording.

    Reproduces SpikeInterface 0.104.3's ``compute_motion``
    (``preprocessing/motion.py:279-461``) step by step with two changes:

    1. ``noise_levels`` is passed explicitly (:func:`estimation_noise_levels`)
       instead of ``compute_motion``'s unseeded ``get_noise_levels`` call
       (``motion.py:359``).
    2. Between localization and estimation, only peaks whose localization
       window lies inside one statistics span are kept
       (:func:`peaks_within_spans`).

    Detection and localization run as ``compute_motion``'s own pipeline for an
    empty ``select_kwargs`` (``motion.py:371-412``): the detector node, an
    ``ExtractDenseWaveforms`` node with the 0.1/0.3 ms window (``motion.py:
    387``) and the localization node, in one ``run_node_pipeline`` pass.
    ``compute_motion``'s other branch (``motion.py:413-432``) is not used: its
    ``localize_peaks`` defaults to a 0.5/0.5 ms window and, for
    ``grid_convolution``, replaces the Gaussian prototype with one built from
    the detected peaks (``sortingcomponents/peak_localization/main.py:44-56``),
    so it would not reproduce what the presets run. Each peak is localized
    from its own waveform, so dropping peaks after localization keeps the
    other peaks' locations unchanged. ``estimate_motion`` then receives every
    resolved argument explicitly. On one span covering the recording with no
    peak dropped, the result equals ``compute_motion`` run with the same
    seeded noise levels.

    Parameters
    ----------
    recording : si.BaseRecording
        Single-segment, unwhitened recording, already silenced over any masked
        ranges. Mutated: its planar geometry is flattened, its time vector
        replaced by the estimation clock (:func:`use_estimation_clock`), and
        its ``noise_level_mad_raw`` property set.
    statistics_spans : array_like
        ``(n, 2)`` half-open frame ranges of valid samples, sorted and
        disjoint, each inside one continuity span.
    continuity_spans : array_like
        ``(m, 2)`` half-open frame ranges of uninterrupted acquisition covering
        the recording. Only ``m == 1`` is supported.
    resolved_params : dict
        Output of :func:`resolve_estimation_params`.
    job_kwargs : dict, optional
        SpikeInterface job kwargs for the detect-and-localize pass. A
        Spyglass ``random_seed`` key is ignored.

    Returns
    -------
    motion : spikeinterface.core.motion.Motion
        The single-segment estimate.
    diagnostics : MotionDiagnostics
        Peak counts and noise levels.

    Raises
    ------
    ValueError
        If the input has more than one continuity span, invalid spans, an
        ineligible geometry (:func:`check_estimation_eligibility`), no kept
        peaks, or a non-finite displacement.
    """
    from spikeinterface.core.job_tools import fix_job_kwargs
    from spikeinterface.core.node_pipeline import (
        ExtractDenseWaveforms,
        run_node_pipeline,
    )
    from spikeinterface.sortingcomponents.motion import estimate_motion
    from spikeinterface.sortingcomponents.peak_detection import (
        detect_peak_methods,
    )
    from spikeinterface.sortingcomponents.peak_localization import (
        peak_localization_methods,
    )

    if recording.get_num_segments() != 1:
        raise ValueError(
            "Motion estimation: expected a single-segment recording; got "
            f"{recording.get_num_segments()} segments."
        )
    n_samples = int(recording.get_num_samples())
    continuity = normalize_spans(continuity_spans)
    statistics = normalize_spans(statistics_spans)
    _check_estimation_spans(n_samples, continuity, statistics)
    check_estimation_eligibility(recording, resolved_params)
    use_estimation_clock(recording)

    noise_levels = estimation_noise_levels(
        recording, statistics, resolved_params["noise_levels_kwargs"]
    )
    detect, localize, estimate = spikeinterface_step_kwargs(resolved_params)
    job_kwargs = fix_job_kwargs(
        {k: v for k, v in (job_kwargs or {}).items() if k != "random_seed"}
    )
    window = resolved_params["localization_window_ms"]

    detect_node = detect_peak_methods[detect.pop("method")](
        recording, noise_levels=noise_levels, **detect
    )
    waveform_node = ExtractDenseWaveforms(
        recording,
        parents=[detect_node],
        ms_before=float(window["ms_before"]),
        ms_after=float(window["ms_after"]),
    )
    localize_node = peak_localization_methods[localize.pop("method")](
        recording,
        parents=[detect_node, waveform_node],
        return_output=True,
        **localize,
    )
    peaks, peak_locations = run_node_pipeline(
        recording,
        [detect_node, waveform_node, localize_node],
        job_kwargs,
        job_name="detect and localize",
        gather_mode="memory",
        gather_kwargs=None,
        squeeze_output=False,
        folder=None,
        names=None,
    )

    keep = peaks_within_spans(
        peaks["sample_index"],
        statistics,
        n_before=waveform_node.nbefore,
        n_after=waveform_node.nafter,
    )
    n_kept = int(keep.sum())
    if n_kept == 0:
        raise ValueError(
            f"Motion estimation: {len(peaks)} peaks were detected but none "
            "has its localization window inside a statistics span; there is "
            "no valid evidence to estimate motion from."
        )
    kept_peaks = peaks[keep]
    motion = estimate_motion(
        recording,
        kept_peaks,
        peak_locations[keep],
        progress_bar=False,
        **estimate,
    )
    displacement = np.asarray(motion.displacement[0])
    if not np.isfinite(displacement).all():
        raise ValueError(
            "Motion estimation: the estimated displacement has "
            f"{int((~np.isfinite(displacement)).sum())} non-finite values."
        )
    # Count against the histogram bins dredge_ap estimated on
    # (``np.arange(t0, t_last + bin_s, bin_s)``, ``motion_utils.py:230-233``),
    # recovered from the bin centers; ``Motion.temporal_bin_edges_s`` instead
    # clips the outer edges to the outer centers (``core/motion.py:302-305``).
    centers = np.asarray(motion.temporal_bins_s[0], dtype=float)
    half_bin = float(estimate["bin_s"]) / 2.0
    edges = np.append(centers - half_bin, centers[-1] + half_bin)
    times = recording.sample_index_to_time(kept_peaks["sample_index"])
    per_bin, _ = np.histogram(times, bins=edges)
    return motion, MotionDiagnostics(
        n_peaks_detected=int(len(peaks)),
        n_peaks_kept=n_kept,
        peaks_per_temporal_bin=per_bin.astype(np.int64),
        noise_levels=noise_levels,
    )


#: Columns the removed in-concat motion correction left on the concat tables.
#: A live heading that still carries one belongs to an un-recreated schema
#: whose cached traces may already be motion corrected.
LEGACY_CONCAT_MOTION_ATTRIBUTES = frozenset(
    {"motion_preset", "motion_correction_params_name"}
)


def assert_concat_schema_current(*heading_names) -> None:
    """Refuse concat tables whose live heading predates motion's removal.

    Parameters
    ----------
    *heading_names : iterable of str
        The live ``heading.names`` of ``ConcatenatedRecording`` and
        ``ConcatenatedRecordingSelection``.

    Raises
    ------
    ValueError
        If any heading still carries a removed motion attribute, naming the
        recreation the user must run first.
    """
    present = set().union(*(set(names) for names in heading_names))
    legacy = sorted(LEGACY_CONCAT_MOTION_ATTRIBUTES & present)
    if legacy:
        raise ValueError(
            "The v2 concat tables still carry the removed motion-correction "
            f"column(s) {legacy}: this database was not recreated after "
            "motion correction moved out of concatenation, so a cached concat "
            "artifact may already be motion corrected and must not be "
            "estimated or corrected again. Recreate the v2 concat tables as "
            "the CHANGELOG describes, then re-populate them."
        )


def motion_estimate_identity_payload(
    *,
    source_kind: str,
    source_id,
    source_content_hash: str,
    artifact_detection_id,
    motion_estimation_params_name: str,
    resolved_params_hash: str,
    spikeinterface_version: str,
    motion_algorithm_version: int,
) -> dict:
    """The logical identity a ``motion_estimate_id`` is derived from.

    ``artifact_detection_id`` is omitted (not encoded as ``null``) when there
    is no artifact pass.

    Parameters
    ----------
    source_kind : {"recording", "concatenated_recording"}
    source_id : uuid.UUID or str
        The ``recording_id`` or ``concat_recording_id``.
    source_content_hash : str
        The source artifact's persisted ``content_hash``.
    artifact_detection_id : uuid.UUID, str or None
    motion_estimation_params_name : str
    resolved_params_hash : str
        :func:`resolved_params_hash` of the resolved configuration.
    spikeinterface_version : str
    motion_algorithm_version : int

    Returns
    -------
    dict
        Payload for ``_selection_identity.deterministic_id``.
    """
    payload = {
        "source_kind": source_kind,
        "source_id": source_id,
        "source_content_hash": source_content_hash,
        "motion_estimation_params_name": motion_estimation_params_name,
        "resolved_params_hash": resolved_params_hash,
        "spikeinterface_version": spikeinterface_version,
        "motion_algorithm_version": int(motion_algorithm_version),
    }
    if artifact_detection_id is not None:
        payload["artifact_detection_id"] = artifact_detection_id
    return payload


def motion_input_fingerprint(
    *,
    source_content_hash: str,
    artifact_detection_id,
    n_samples: int,
    sampling_frequency: float,
    continuity_spans,
    statistics_spans,
    channel_ids,
    channel_locations,
    resolved_params_hash: str,
) -> str:
    """SHA-256 of everything an estimate was computed from.

    The source content, mask choice, frame spans, estimation channels and
    their positions, and the resolved configuration. Two estimates with the
    same fingerprint read the same valid samples on the same geometry with
    the same settings.

    Returns
    -------
    str
        64-character hex digest.
    """
    payload = {
        "source_content_hash": str(source_content_hash),
        "artifact_detection_id": (
            None
            if artifact_detection_id is None
            else str(artifact_detection_id)
        ),
        "n_samples": int(n_samples),
        "sampling_frequency": float(sampling_frequency),
        "continuity_spans": [
            list(span) for span in normalize_spans(continuity_spans)
        ],
        "statistics_spans": [
            list(span) for span in normalize_spans(statistics_spans)
        ],
        "channel_ids": [str(c) for c in np.asarray(channel_ids).tolist()],
        "channel_locations": np.asarray(
            channel_locations, dtype=float
        ).tolist(),
        "resolved_params_hash": str(resolved_params_hash),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
