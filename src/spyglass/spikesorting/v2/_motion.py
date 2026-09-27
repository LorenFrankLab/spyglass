"""SpikeInterface motion estimation: parameters, serialization, summaries.

``motion_to_storage_dict`` / ``motion_from_storage_dict`` round-trip a
``Motion`` through a DataJoint-blob-safe dict; ``motion_max_abs_displacement_um``
and ``motion_n_temporal_bins`` summarize one for a stored row.

``resolve_estimation_params`` turns a ``MotionEstimationParameters`` blob into
the fully resolved configuration the estimator passes to SpikeInterface, and
``resolved_params_hash`` content-addresses it.

``build_estimation_clock`` places a source's continuity spans on one
*estimation clock* (real gaps between spans kept up to a cap), and
``EstimationClockRecording`` presents a recording on that clock, so one
estimation covers every span in a single reference frame.

``estimate_motion_in_spans`` estimates motion from the valid samples of one
recording: it reproduces SpikeInterface's ``compute_motion`` with explicit
noise levels from the statistics spans and keeps only peaks whose localization
window lies inside one statistics span.

``resolve_interpolation_params`` canonicalizes a
``MotionInterpolationParameters`` blob, and
``apply_motion_on_estimation_clock`` interpolates a saved estimate onto the
recording it was estimated from, on the same estimation clock.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection. Only ``spikeinterface.core`` is imported at module level (the
clock view subclasses its recording classes); the rest of SpikeInterface is
imported inside the functions that need it.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from typing import NamedTuple

import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment

from spyglass.spikesorting.v2._sorting_dispatch import (
    STATISTICS_SAMPLE_CHUNK_MS,
    STATISTICS_SAMPLE_NUM_CHUNKS,
)

#: Version of the motion-estimation algorithm this module implements (the
#: input calibration and masking, SpikeInterface call sequence, peak filter,
#: noise estimate and resolution rules). Part of every estimate's identity;
#: bump it when any of those change in a way that can change a stored
#: estimate.
MOTION_ALGORITHM_VERSION = 3

#: Version of the motion-application algorithm
#: (:func:`apply_motion_on_estimation_clock`: masking, clock, interpolation
#: call). Part of every corrected recording's identity; bump it when a change
#: can change stored corrected traces.
MOTION_INTERPOLATION_ALGORITHM_VERSION = 2

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

#: Random-chunk budget of the noise estimate, taken from the span sampler
#: (``_sorting_dispatch._sample_statistics_spans``) so the resolved
#: configuration -- and hence the estimate's identity -- follows any change
#: to that sampler.
_NOISE_METHOD = "mad"
_NOISE_NUM_CHUNKS = STATISTICS_SAMPLE_NUM_CHUNKS
_NOISE_CHUNK_DURATION = f"{STATISTICS_SAMPLE_CHUNK_MS}ms"


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
    - ``max_gap_s``: the estimation clock's cap on the time between continuity
      spans (:func:`build_estimation_clock`).

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
        "max_gap_s": float(validated.max_gap_s),
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
        Peaks whose localization window lies inside one statistics span and
        whose detection window does not cross a continuity-span join; only
        these reach the estimator.
    peaks_per_temporal_bin : numpy.ndarray
        ``(n_temporal_bins,)`` int64 count of kept peaks in each of the
        estimate's temporal bins.
    peaks_per_continuity_span : numpy.ndarray
        ``(n_spans,)`` int64 count of kept peaks in each continuity span; a
        zero marks a span that contributed no evidence.
    noise_levels : numpy.ndarray
        ``(n_channels,)`` float64 per-channel noise (recording units) the
        detection threshold was scaled by.
    """

    n_peaks_detected: int
    n_peaks_kept: int
    peaks_per_temporal_bin: np.ndarray
    peaks_per_continuity_span: np.ndarray
    noise_levels: np.ndarray


def normalize_spans(spans) -> list[tuple[int, int]]:
    """Return ``spans`` as a list of ``(start, end)`` Python-int tuples."""
    return [
        (int(a), int(b))
        for a, b in np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    ]


def _check_estimation_spans(
    n_samples: int,
    clock: EstimationClock,
    statistics_spans: list[tuple[int, int]],
) -> None:
    """Validate the span inputs of :func:`estimate_motion_in_spans`.

    Raises
    ------
    ValueError
        If the clock's continuity spans do not cover the recording, or the
        statistics spans are empty, unsorted, overlapping, outside the
        recording or cross a continuity-span edge.
    """
    if int(clock.spans[-1, 1]) != int(n_samples):
        raise ValueError(
            f"Motion estimation: the continuity spans {clock.spans.tolist()} "
            f"do not cover the recording's {n_samples} samples."
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
    starts = [a for a, _ in statistics_spans]
    span = np.searchsorted(clock.spans[:, 0], starts, side="right") - 1
    ends = np.array([b for _, b in statistics_spans])
    crossing = np.flatnonzero(ends > clock.spans[span, 1])
    if crossing.size:
        raise ValueError(
            "Motion estimation: statistics spans "
            f"{[statistics_spans[i] for i in crossing]} cross a continuity "
            f"span edge ({clock.spans.tolist()})."
        )


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


def peaks_clear_of_joins(
    sample_index, continuity_spans, *, margin: int
) -> np.ndarray:
    """Mark peaks whose detection did not read across an internal join.

    SpikeInterface's ``locally_exclusive`` detector drops a peak when a larger
    one on a neighbouring channel lies within ``exclude_sweep_size`` frames
    (``sortingcomponents/peak_detection/locally_exclusive.py:150-183``), and a
    peak is itself a local extremum of its neighbouring frames (``:127-143``),
    so whether a peak at frame ``s`` is detected depends on frames
    ``[s - margin, s + margin]`` with ``margin = exclude_sweep_size + 1``, the
    detector's ``get_trace_margin`` (``:86-88``). Across a boundary between
    continuity spans (an acquisition gap or a concatenation join) those frames
    are adjacent in the recording but not in time, so a peak on one side can
    suppress a peak on the other. A peak is kept only when that window stays
    inside its own continuity span. The recording's first start and last end
    are not joins (the detector reads no frames past them), so a single span
    keeps every peak.

    Parameters
    ----------
    sample_index : numpy.ndarray
        ``(n_peaks,)`` peak frames.
    continuity_spans : numpy.ndarray
        ``(n_spans, 2)`` contiguous half-open continuity spans covering the
        recording from frame 0.
    margin : int
        Keyword-only. Frames on each side of a peak its detection reads.

    Returns
    -------
    numpy.ndarray
        ``(n_peaks,)`` bool mask of kept peaks.
    """
    sample_index = np.asarray(sample_index, dtype=np.int64)
    spans = np.asarray(continuity_spans, dtype=np.int64)
    span = np.searchsorted(spans[:, 0], sample_index, side="right") - 1
    return ((span == 0) | (sample_index - margin >= spans[span, 0])) & (
        (span == len(spans) - 1) | (sample_index + margin < spans[span, 1])
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


def unfiltered_source_problem(
    preprocessing_params_name: str, preprocessing_params: dict
) -> "str | None":
    """Say why a source's preprocessing recipe cannot feed motion estimation.

    Motion is estimated on filtered, unwhitened traces: peaks are detected
    where a trace crosses a multiple of its noise level, and localized from
    their waveforms, which assumes the traces carry no DC or slow drift of
    their own. A recipe with ``bandpass_filter=None`` (the shipped
    ``no_filter`` row) applies no temporal filter at all, so its traces keep
    the acquisition's DC and low-frequency content.

    Parameters
    ----------
    preprocessing_params_name : str
        The source's ``PreprocessingParameters`` row (for a concatenated
        recording, the concatenation's recipe).
    preprocessing_params : dict
        That row's ``params`` blob.

    Returns
    -------
    str or None
        The problem, naming the recipe, or ``None`` when the recipe filters.
    """
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )

    params = PreprocessingParamsSchema.model_validate(preprocessing_params)
    if params.bandpass_filter is not None:
        return None
    return (
        f"preprocessing recipe {preprocessing_params_name!r} applies no "
        "temporal filter (bandpass_filter is None). Motion is estimated on "
        "filtered, unwhitened traces: peak detection and localization assume "
        "traces without DC or slow drift of their own. Estimate on a "
        "recording built with a bandpass-filtering recipe (for example "
        "'default')."
    )


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
        The masked microvolt recording. Its ``noise_level_mad_raw`` property
        is set.
    statistics_spans : list[tuple[int, int]]
        Half-open frame spans of valid samples.
    noise_kwargs : dict
        ``resolved_params["noise_levels_kwargs"]``.

    Returns
    -------
    numpy.ndarray
        ``(n_channels,)`` float64 noise levels in the units of
        ``recording``'s traces (microvolts).

    Raises
    ------
    ValueError
        If any channel's noise level is non-finite or not positive (a dead,
        flat or NaN channel): the detection threshold is a multiple of it.
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
            progress_bar=False,
        )
    levels = np.asarray(levels, dtype=np.float64)
    bad = ~(np.isfinite(levels) & (levels > 0))
    if bad.any():
        raise ValueError(
            "Motion estimation: channel(s) "
            f"{np.asarray(recording.channel_ids)[bad].tolist()} have a "
            "non-finite or non-positive noise level "
            f"({levels[bad].tolist()}); the detection threshold is a "
            "multiple of it. Exclude dead, flat or NaN channels from the "
            "source recording before estimating motion."
        )
    return levels


class EstimationClock(NamedTuple):
    """Map from a source's frames to its motion-estimation clock.

    Within continuity span ``i`` (frames ``[a_i, b_i)``), estimation time
    advances by exactly ``1 / fs`` per frame from ``e_i``:
    ``time(s) = e_i + (s - a_i) / fs``. ``e_0 = t_0``, and each later span
    starts after the previous one's nominal duration plus the real gap
    between them, capped (:func:`build_estimation_clock`).

    Attributes
    ----------
    spans : numpy.ndarray
        ``(n_spans, 2)`` int64 half-open continuity spans ``[a_i, b_i)``,
        contiguous and covering the recording from frame 0.
    source_start_s : numpy.ndarray
        ``(n_spans,)`` float64 ``t_i``: each span's first timestamp on the
        source's own acquisition clock (s).
    source_end_s : numpy.ndarray
        ``(n_spans,)`` float64: each span's last timestamp on that clock (s).
    estimation_start_s : numpy.ndarray
        ``(n_spans,)`` float64 ``e_i``: each span's start on the estimation
        clock (s).
    sampling_frequency : float
        ``fs`` (Hz).
    """

    spans: np.ndarray
    source_start_s: np.ndarray
    source_end_s: np.ndarray
    estimation_start_s: np.ndarray
    sampling_frequency: float


def build_estimation_clock(
    spans,
    source_start_s,
    source_end_s,
    sampling_frequency: float,
    max_gap_s: float,
) -> EstimationClock:
    """Place continuity spans on one estimation clock.

    ``e_0 = t_0`` and ``e_{i+1} = e_i + (b_i - a_i) / fs + min(g_i,
    max_gap_s)``, where ``g_i = t_{i+1} - (u_i + 1 / fs)`` is the real gap
    after span ``i``: from one sample after its last timestamp ``u_i`` to the
    next span's first timestamp. Inside a span the clock uses the nominal
    duration ``(b_i - a_i) / fs``; the gap is measured on the real timestamps
    because a span whose timestamps run slightly off ``fs`` (SpikeInterface
    derives ``fs`` from a timestamped series' first 1000 steps,
    ``extractors/nwbextractors.py:369``) drifts from its nominal end by
    ``ppm * duration``, which would otherwise swamp a gap of a few dropped
    frames. A next span starting after the last timestamp but less than one
    sample after it is adjacent (``g_i`` is clamped to 0): float rounding of
    the timestamps and of ``fs`` makes a true zero gap come out slightly
    negative. A gap longer than ``max_gap_s`` is shortened to it; a shorter
    gap keeps its real length. Every span thus shares one clock (and one
    estimation), while the unobserved time between spans stays bounded: the
    estimator's temporal bins, and its dense bin-by-bin correlation matrices,
    grow with the clock's total length.

    Parameters
    ----------
    spans : array_like
        ``(n_spans, 2)`` half-open continuity spans in frames, contiguous and
        starting at frame 0.
    source_start_s : array_like
        ``(n_spans,)`` first timestamp ``t_i`` of each span (s).
    source_end_s : array_like
        ``(n_spans,)`` last timestamp ``u_i`` of each span (s).
    sampling_frequency : float
        ``fs`` (Hz).
    max_gap_s : float
        The cap on each gap (s), ``>= 0``.

    Returns
    -------
    EstimationClock

    Raises
    ------
    ValueError
        If the spans are empty, not contiguous from frame 0, or empty spans;
        if a timestamp or the cap is not finite (or the cap is negative); if a
        span ends before it starts; or if a span starts at or before the
        previous span's last timestamp (overlapping or out-of-order spans).
    """
    spans = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    starts = np.asarray(source_start_s, dtype=np.float64).reshape(-1)
    ends = np.asarray(source_end_s, dtype=np.float64).reshape(-1)
    fs = float(sampling_frequency)
    max_gap_s = float(max_gap_s)
    if len(spans) == 0:
        raise ValueError("Estimation clock: no continuity spans were given.")
    if len(starts) != len(spans) or len(ends) != len(spans):
        raise ValueError(
            f"Estimation clock: {len(spans)} continuity spans but "
            f"{len(starts)} start times and {len(ends)} end times."
        )
    if (
        spans[0, 0] != 0
        or np.any(spans[:, 1] <= spans[:, 0])
        or np.any(spans[1:, 0] != spans[:-1, 1])
    ):
        raise ValueError(
            "Estimation clock: continuity spans must be non-empty and "
            f"contiguous from frame 0; got {spans.tolist()}."
        )
    if not (np.isfinite(starts).all() and np.isfinite(ends).all()):
        raise ValueError(
            "Estimation clock: span start and end times must be finite; got "
            f"{starts.tolist()} and {ends.tolist()}."
        )
    if np.any(ends < starts):
        raise ValueError(
            "Estimation clock: a span's last timestamp precedes its first; "
            f"got starts {starts.tolist()} and ends {ends.tolist()}."
        )
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError(
            f"Estimation clock: sampling_frequency must be positive; got {fs}."
        )
    if not (np.isfinite(max_gap_s) and max_gap_s >= 0):
        raise ValueError(
            f"Estimation clock: max_gap_s must be finite and >= 0; got "
            f"{max_gap_s}."
        )
    estimation = [float(starts[0])]
    for i in range(len(spans) - 1):
        duration = (spans[i, 1] - spans[i, 0]) / fs
        gap = starts[i + 1] - (ends[i] + 1.0 / fs)
        # Adjacent spans (a member join with no real gap) can compute a gap a
        # rounding error below zero: the timestamps, and fs itself, are
        # floats. Only a next start at or before the last timestamp is an
        # overlap; anything between is a zero gap.
        if starts[i + 1] <= ends[i]:
            raise ValueError(
                f"Estimation clock: continuity span {i + 1} (frames "
                f"{spans[i + 1].tolist()}) starts at {starts[i + 1]!r} s, "
                f"at or before span {i}'s last timestamp ({ends[i]!r} s; "
                f"frames {spans[i].tolist()}). Motion estimation places all "
                "spans on one acquisition clock, so their timestamps must "
                "increase across spans. For a concatenation, the members' "
                "clocks may be independent (for example each starting at 0 "
                "in a different NWB file) or the members may be out of "
                "acquisition order; concatenate members that share one "
                "clock, in acquisition order."
            )
        gap = max(gap, 0.0)
        estimation.append(estimation[-1] + duration + min(gap, max_gap_s))
    return EstimationClock(
        spans=spans,
        source_start_s=starts,
        source_end_s=ends,
        estimation_start_s=np.asarray(estimation, dtype=np.float64),
        sampling_frequency=fs,
    )


def estimation_clock_from_blob(blob: dict) -> EstimationClock:
    """Rebuild an :class:`EstimationClock` from its stored ``_asdict()``."""
    return EstimationClock(
        spans=np.asarray(blob["spans"], dtype=np.int64).reshape(-1, 2),
        source_start_s=np.asarray(
            blob["source_start_s"], dtype=np.float64
        ).reshape(-1),
        source_end_s=np.asarray(blob["source_end_s"], dtype=np.float64).reshape(
            -1
        ),
        estimation_start_s=np.asarray(
            blob["estimation_start_s"], dtype=np.float64
        ).reshape(-1),
        sampling_frequency=float(blob["sampling_frequency"]),
    )


def estimation_times(clock: EstimationClock, sample_index):
    """Estimation-clock times (s) of source frames.

    ``e_i + (s - a_i) / fs`` for the span ``i`` holding frame ``s``. With one
    span starting at frame 0 this is ``s / fs + t_0``, the arithmetic
    SpikeInterface's own time-vector-free clock uses
    (``core/baserecording.py:975-985``). Frames before 0 or past the end are
    extrapolated from the first / last span.

    Parameters
    ----------
    clock : EstimationClock
    sample_index : int or array_like of int
        Source frame(s).

    Returns
    -------
    float or numpy.ndarray
        Time(s) in seconds, the shape of ``sample_index``.
    """
    sample_index = np.asarray(sample_index)
    starts = clock.spans[:, 0]
    span = np.clip(
        np.searchsorted(starts, sample_index, side="right") - 1, 0, None
    )
    return (sample_index - starts[span]) / clock.sampling_frequency + (
        clock.estimation_start_s[span]
    )


class SourceClockDisplacement(NamedTuple):
    """A motion estimate's temporal bins mapped back to the source clock.

    Attributes
    ----------
    source_time_s : numpy.ndarray
        ``(n_temporal_bins,)`` float64 source-clock time of each bin center
        (span ``i``'s estimation interval mapped affinely onto its real
        extent, :func:`displacement_on_source_clock`); NaN for a bin whose
        center lies in a capped gap.
    continuity_span : numpy.ndarray
        ``(n_temporal_bins,)`` int64 continuity span holding each bin center;
        ``-1`` for a bin inside a capped gap.
    in_gap : numpy.ndarray
        ``(n_temporal_bins,)`` bool, ``continuity_span == -1``.
    displacement_um : numpy.ndarray
        ``(n_temporal_bins, n_spatial_bins)`` estimated displacement (um).
    spatial_bins_um : numpy.ndarray
        ``(n_spatial_bins,)`` window centers (um).
    """

    source_time_s: np.ndarray
    continuity_span: np.ndarray
    in_gap: np.ndarray
    displacement_um: np.ndarray
    spatial_bins_um: np.ndarray


def displacement_on_source_clock(
    motion, clock: EstimationClock
) -> SourceClockDisplacement:
    """Map a single-segment estimate's temporal bins to source time.

    A bin belongs to the continuity span whose estimation-clock interval
    ``[e_i, e_i + (b_i - a_i) / fs)`` holds its center. That interval is
    mapped affinely onto the span's real extent ``[t_i, u_i + 1 / fs)``
    (first timestamp to one sample after the last), so the source time is
    ``t_i + (center - e_i) * (u_i + 1 / fs - t_i) / ((b_i - a_i) / fs)``:
    when the timestamps run at a rate other than the nominal ``fs`` (which
    SpikeInterface derives from a series' first 1000 steps), inspection
    times do not inherit that scale error. A center between two spans lies
    in a capped gap and is reported there, not assigned to a span. A center
    before the first span or past the last span's end belongs to that span
    (extrapolated): those are the outer histogram bins that hold its first /
    final samples.

    Parameters
    ----------
    motion : spikeinterface.core.motion.Motion
        A single-segment estimate whose bins are on ``clock``.
    clock : EstimationClock

    Returns
    -------
    SourceClockDisplacement
    """
    centers = np.asarray(motion.temporal_bins_s[0], dtype=np.float64)
    fs = clock.sampling_frequency
    starts = clock.estimation_start_s
    nominal = (clock.spans[:, 1] - clock.spans[:, 0]) / fs
    real = clock.source_end_s + 1.0 / fs - clock.source_start_s
    ends = starts + nominal
    span = np.clip(np.searchsorted(starts, centers, side="right") - 1, 0, None)
    in_gap = (centers >= ends[span]) & (span < len(starts) - 1)
    source = clock.source_start_s[span] + (centers - starts[span]) * (
        real[span] / nominal[span]
    )
    source[in_gap] = np.nan
    return SourceClockDisplacement(
        source_time_s=source,
        continuity_span=np.where(in_gap, -1, span).astype(np.int64),
        in_gap=in_gap,
        displacement_um=np.asarray(motion.displacement[0]),
        spatial_bins_um=np.asarray(motion.spatial_bins_um),
    )


class EstimationClockRecordingSegment(BaseRecordingSegment):
    """One recording segment presented on an estimation clock."""

    def __init__(self, parent_segment, clock: EstimationClock):
        BaseRecordingSegment.__init__(
            self, sampling_frequency=clock.sampling_frequency
        )
        self._parent_segment = parent_segment
        self._clock = clock

    def get_num_samples(self) -> int:
        return self._parent_segment.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        return self._parent_segment.get_traces(
            start_frame, end_frame, channel_indices
        )

    def sample_index_to_time(self, sample_ind):
        return estimation_times(self._clock, sample_ind)

    def time_to_sample_index(self, time_s):
        """The frame nearest ``time_s`` within its span on the estimation clock.

        Within a span this rounds like SpikeInterface's time-vector-free
        lookup (``core/baserecording.py:987-993``); a time inside a capped gap
        maps to the last frame of the span before it.
        """
        time_s = np.asarray(time_s, dtype=np.float64)
        clock = self._clock
        span = np.clip(
            np.searchsorted(clock.estimation_start_s, time_s, side="right") - 1,
            0,
            None,
        )
        frame = clock.spans[span, 0] + np.round(
            (time_s - clock.estimation_start_s[span]) * clock.sampling_frequency
        ).astype(np.int64)
        return np.minimum(frame, clock.spans[span, 1] - 1)

    def get_times(self) -> np.ndarray:
        return estimation_times(
            self._clock, np.arange(self.get_num_samples(), dtype=np.int64)
        )

    def get_start_time(self) -> float:
        return float(self._clock.estimation_start_s[0])

    def get_end_time(self) -> float:
        return float(estimation_times(self._clock, self.get_num_samples() - 1))


class EstimationClockRecording(BaseRecording):
    """A single-segment recording presented on its estimation clock.

    Traces, channels, properties and probe are the parent's; only the time
    lookups follow the :class:`EstimationClock`, computed per call from the
    span table, so no ``n_samples`` time vector is materialized (a float64
    vector would cost 8 bytes per frame, about 0.86 GB per hour at 30 kHz).
    SpikeInterface's motion estimator reads peak times through
    ``recording.sample_index_to_time`` (``sortingcomponents/motion/dredge.py:
    227``, ``motion_utils.py:230-236``), and the parent's HDF5-backed NWB time
    vector is never indexed, so repeated peak frames are safe.

    Preprocessors built on top of this view see a plain ``1 / fs`` clock of
    their own; only code that asks this view (or its segment) for times sees
    the estimation clock.

    Parameters
    ----------
    recording : si.BaseRecording
        Single-segment parent recording.
    clock : EstimationClock or dict
        Its clock (or the clock's ``_asdict()``, as SpikeInterface passes it
        back when it rebuilds the view from its kwargs); the spans must cover
        the recording and ``fs`` must equal the recording's.
    """

    def __init__(self, recording, clock: EstimationClock | dict):
        if not isinstance(clock, EstimationClock):
            clock = estimation_clock_from_blob(clock)
        if recording.get_num_segments() != 1:
            raise ValueError(
                "EstimationClockRecording: expected a single-segment "
                f"recording; got {recording.get_num_segments()} segments."
            )
        n_samples = int(recording.get_num_samples())
        if int(clock.spans[-1, 1]) != n_samples:
            raise ValueError(
                f"EstimationClockRecording: the clock's spans end at frame "
                f"{int(clock.spans[-1, 1])}, but the recording has "
                f"{n_samples} samples."
            )
        if float(clock.sampling_frequency) != float(
            recording.get_sampling_frequency()
        ):
            raise ValueError(
                "EstimationClockRecording: the clock's sampling frequency "
                f"{clock.sampling_frequency!r} differs from the recording's "
                f"{recording.get_sampling_frequency()!r}."
            )
        BaseRecording.__init__(
            self,
            recording.get_sampling_frequency(),
            recording.channel_ids,
            recording.get_dtype(),
        )
        recording.copy_metadata(self, only_main=False)
        self.add_recording_segment(
            EstimationClockRecordingSegment(
                recording._recording_segments[0], clock
            )
        )
        self._serializability["json"] = False
        self._kwargs = {
            "recording": recording,
            "clock": {
                key: (
                    value.tolist() if isinstance(value, np.ndarray) else value
                )
                for key, value in clock._asdict().items()
            },
        }


def estimate_motion_in_spans(
    recording,
    *,
    statistics_spans,
    clock: EstimationClock,
    resolved_params: dict,
    job_kwargs: dict | None = None,
):
    """Estimate motion from the valid samples of one recording, once.

    Reproduces SpikeInterface 0.104.3's ``compute_motion``
    (``preprocessing/motion.py:279-461``) step by step with four changes:

    1. Every step reads the recording as float microvolts
       (:func:`recording_in_microvolts`) silenced outside the statistics
       spans, so the detection threshold and the masked zeros are the same
       physical voltages whatever the source's dtype, gains and offsets. A
       unit-calibrated float recording with one span covering it is used
       as is.
    2. ``noise_levels`` is passed explicitly (:func:`estimation_noise_levels`)
       instead of ``compute_motion``'s unseeded ``get_noise_levels`` call
       (``motion.py:359``).
    3. Between localization and estimation, only peaks whose localization
       window lies inside one statistics span (:func:`peaks_within_spans`)
       and whose detection did not read across a join between continuity
       spans (:func:`peaks_clear_of_joins`) are kept.
    4. ``estimate_motion`` reads peak times on the estimation clock
       (:class:`EstimationClockRecording`), so every continuity span is
       estimated in one call and shares one reference frame. DREDge centres
       its displacement on the bins that hold data (``sortingcomponents/
       motion/dredge.py:853-870``), so estimating spans separately would give
       each span its own arbitrary offset.

    Detection and localization run as ``compute_motion``'s own pipeline for an
    empty ``select_kwargs`` (``motion.py:373-412``): the detector node, an
    ``ExtractDenseWaveforms`` node with the 0.1/0.3 ms window (``motion.py:
    387``) and the localization node, in one ``run_node_pipeline`` pass on
    the microvolt view itself (these steps read traces, never times).
    ``compute_motion``'s other branch (``motion.py:413-432``) is not used: its
    ``localize_peaks`` defaults to a 0.5/0.5 ms window and, for
    ``grid_convolution``, replaces the Gaussian prototype with one built from
    the detected peaks (``sortingcomponents/peak_localization/main.py:44-56``),
    so it would not reproduce what the presets run. Each peak is localized
    from its own waveform, so dropping peaks after localization keeps the
    other peaks' locations unchanged. ``estimate_motion`` then receives every
    resolved argument explicitly. On one span covering the recording with no
    peak dropped, the result equals ``compute_motion`` run with the same
    seeded noise levels on a recording whose clock is ``t_0 + i / fs``.

    Parameters
    ----------
    recording : si.BaseRecording
        Single-segment, unwhitened recording. Samples outside
        ``statistics_spans`` are silenced here, after conversion to
        microvolts. Mutated: its planar geometry is flattened, and its
        ``noise_level_mad_raw`` property is set when it is used as is. Its own
        time vector is not read.
    statistics_spans : array_like
        ``(n, 2)`` half-open frame ranges of valid samples, sorted and
        disjoint, each inside one continuity span.
    clock : EstimationClock
        The recording's continuity spans on the estimation clock
        (:func:`build_estimation_clock`).
    resolved_params : dict
        Output of :func:`resolve_estimation_params`.
    job_kwargs : dict, optional
        SpikeInterface job kwargs for the detect-and-localize pass. A
        Spyglass ``random_seed`` key is ignored.

    Returns
    -------
    motion : spikeinterface.core.motion.Motion
        The single-segment estimate, its temporal bins on the estimation
        clock.
    diagnostics : MotionDiagnostics
        Peak counts and noise levels.

    Raises
    ------
    ValueError
        If the spans are invalid, the geometry is ineligible
        (:func:`check_estimation_eligibility`), no continuity span keeps a
        peak, or the displacement is non-finite. A continuity span without
        kept peaks is only logged and counted in the diagnostics.
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

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        complement_frame_ranges,
        silence_frame_ranges,
    )
    from spyglass.utils import logger

    if recording.get_num_segments() != 1:
        raise ValueError(
            "Motion estimation: expected a single-segment recording; got "
            f"{recording.get_num_segments()} segments."
        )
    n_samples = int(recording.get_num_samples())
    statistics = normalize_spans(statistics_spans)
    _check_estimation_spans(n_samples, clock, statistics)
    check_estimation_eligibility(recording, resolved_params)
    # Detection thresholds traces against a multiple of their noise and
    # silenced samples are zeros: both are physical only on microvolts with
    # zero offset, so calibrate first and silence the calibrated traces.
    recording = silence_frame_ranges(
        recording_in_microvolts(recording),
        complement_frame_ranges(statistics, n_samples),
    )
    clocked = EstimationClockRecording(recording, clock)

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
    ) & peaks_clear_of_joins(
        peaks["sample_index"],
        clock.spans,
        margin=int(detect_node.get_trace_margin()),
    )
    n_kept = int(keep.sum())
    if n_kept == 0:
        raise ValueError(
            f"Motion estimation: {len(peaks)} peaks were detected but none "
            "has its localization window inside a statistics span and its "
            "detection window clear of the continuity-span joins; there is "
            "no valid evidence to estimate motion from."
        )
    kept_peaks = peaks[keep]
    per_span = np.bincount(
        np.searchsorted(
            clock.spans[:, 0], kept_peaks["sample_index"], side="right"
        )
        - 1,
        minlength=len(clock.spans),
    ).astype(np.int64)
    empty = np.flatnonzero(per_span == 0)
    if empty.size:
        logger.warning(
            "Motion estimation: continuity span(s) %s (frames %s) kept no "
            "peaks; the estimate there rests on the temporal prior only.",
            empty.tolist(),
            clock.spans[empty].tolist(),
        )
    motion = estimate_motion(
        clocked,
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
    times = clocked.sample_index_to_time(kept_peaks["sample_index"])
    per_bin, _ = np.histogram(times, bins=edges)
    return motion, MotionDiagnostics(
        n_peaks_detected=int(len(peaks)),
        n_peaks_kept=n_kept,
        peaks_per_temporal_bin=per_bin.astype(np.int64),
        peaks_per_continuity_span=per_span,
        noise_levels=noise_levels,
    )


def resolve_interpolation_params(params: dict) -> dict:
    """Return the motion-interpolation configuration passed to SpikeInterface.

    Every field of :class:`MotionInterpolationParamsSchema` is required, so
    resolution is validation plus a canonical encoding (``p`` and
    ``num_closest`` as ints, ``sigma_um`` as a float) that
    :func:`resolved_params_hash` content-addresses.

    Parameters
    ----------
    params : dict
        A ``MotionInterpolationParameters.params`` blob (validated or as
        fetched).

    Returns
    -------
    dict
        ``border_mode``, ``spatial_interpolation_method``, ``sigma_um``, ``p``
        and ``num_closest``.
    """
    from spyglass.spikesorting.v2._lookup_validation import _jsonable_blob
    from spyglass.spikesorting.v2._params.motion_interpolation import (
        MotionInterpolationParamsSchema,
    )

    validated = MotionInterpolationParamsSchema.model_validate(
        _jsonable_blob(params)
    )
    return {
        "border_mode": str(validated.border_mode),
        "spatial_interpolation_method": str(
            validated.spatial_interpolation_method
        ),
        "sigma_um": float(validated.sigma_um),
        "p": int(validated.p),
        "num_closest": int(validated.num_closest),
    }


class AppliedMotion(NamedTuple):
    """A lazily motion-corrected recording and the channels it dropped.

    Attributes
    ----------
    recording : si.BaseRecording
        The corrected, masked recording; its channel locations are the
        source's unmoved positions of the kept channels.
    removed_channel_ids : list
        Source channel ids that ``remove_channels`` dropped, in source order;
        empty for ``force_extrapolate``.
    """

    recording: BaseRecording
    removed_channel_ids: list


def recording_in_microvolts(recording):
    """Present a recording as float microvolts with a unit calibration.

    Estimation thresholds each trace against a multiple of its noise and
    treats silenced samples as zeros, which is physical only for traces in
    microvolts with zero offset: on raw counts with a nonzero offset, the
    offset moves every sample away from the threshold's zero and a silenced
    sample reads as the offset voltage.

    Interpolation mixes channels with weights that need not sum to 1 (the
    extrapolated contacts at the probe's ends), so it must act on physical
    voltages: interpolating raw counts and keeping the source's gains and
    offsets changes the physical value of every weighted sum whose weights do
    not sum to 1, and of any mix of channels with different gains.

    A float recording whose gains are all 1 and offsets all 0 (or that has
    neither, a synthetic one) is already in its physical units and is
    returned unchanged. Any other recording goes through SpikeInterface's
    ``scale_to_uV`` (``preprocessing/scale.py:68-102``), which computes
    ``raw * gain + offset`` per channel in float32 and sets gains 1 and
    offsets 0 (``preprocessing/normalize_scale.py:21-34``).

    Parameters
    ----------
    recording : si.BaseRecording
        The source recording.

    Returns
    -------
    si.BaseRecording
        ``recording`` itself, or its float32 microvolt view.

    Raises
    ------
    RuntimeError
        From ``scale_to_uV`` if a recording that needs scaling lacks gains or
        offsets.
    """
    import spikeinterface.preprocessing as sip

    gains = recording.get_channel_gains()
    offsets = recording.get_channel_offsets()
    if gains is None and offsets is None:
        unit = True
    elif gains is None or offsets is None:
        unit = False
    else:
        unit = bool(np.all(gains == 1) and np.all(offsets == 0))
    if unit and recording.get_dtype().kind == "f":
        return recording
    return sip.scale_to_uV(recording)


def apply_motion_on_estimation_clock(
    recording,
    motion,
    *,
    clock: EstimationClock,
    statistics_spans,
    resolved_interpolation: dict,
) -> AppliedMotion:
    """Interpolate a saved motion estimate onto the recording it came from.

    The recording is silenced outside its statistics spans (the samples the
    estimate treated as masked), presented on the estimate's estimation clock
    (:class:`EstimationClockRecording`), interpolated with SpikeInterface's
    ``interpolate_motion`` and silenced again. The clock view is the direct
    parent of the ``InterpolateMotionRecording``, whose segments read each
    frame's time from their parent segment
    (``sortingcomponents/motion/motion_interpolation.py:504``) and bin it
    against the motion's temporal bins (``:175-180``): every frame therefore
    looks up the displacement at exactly the time it had during estimation.
    A preprocessor between the two would present a plain ``1 / fs`` clock
    (``preprocessing/basepreprocessor.py:27-29``) and silently shift the
    lookups after the first acquisition gap.

    The recording is first converted to microvolts
    (:func:`recording_in_microvolts`), so the corrected traces are float
    microvolts with gains 1 and offsets 0. Interpolation is frame-local (one
    kernel per temporal bin applied to each frame), so a silenced frame stays
    exactly zero and the sample count, frame order and sampling frequency are
    unchanged; the second silencing states that contract rather than relying
    on it. Every interpolation
    argument is passed explicitly from ``resolved_interpolation``; the
    motion's own temporal bins are the interpolation bins. The output's
    channel locations are the unmoved positions of the kept channels
    (``BasePreprocessor`` copies the parent's metadata).

    Parameters
    ----------
    recording : si.BaseRecording
        Single-segment, unwhitened source recording (the estimate's source
        artifact as read back). Mutated only by planar flattening.
    motion : spikeinterface.core.motion.Motion
        Single-segment estimate whose temporal bins are on ``clock``.
    clock : EstimationClock
        The estimate's time map.
    statistics_spans : array_like
        ``(n, 2)`` half-open frame ranges the estimate treated as valid
        samples; every other frame is silenced.
    resolved_interpolation : dict
        Output of :func:`resolve_interpolation_params`.

    Returns
    -------
    AppliedMotion

    Raises
    ------
    ValueError
        If the recording or motion is not single-segment, the clock does not
        cover the recording, every channel is removed, or the output contact
        positions are not finite and distinct.
    """
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    from spyglass.spikesorting.v2._recording_geometry import (
        flatten_planar_geometry,
    )
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        complement_frame_ranges,
        silence_frame_ranges,
    )

    if recording.get_num_segments() != 1 or motion.num_segments != 1:
        raise ValueError(
            "Motion correction: expected a single-segment recording and "
            f"motion; got {recording.get_num_segments()} and "
            f"{motion.num_segments} segments."
        )
    n_samples = int(recording.get_num_samples())
    flatten_planar_geometry(recording)
    excluded = complement_frame_ranges(
        normalize_spans(statistics_spans), n_samples
    )
    in_uv = recording_in_microvolts(recording)
    masked = silence_frame_ranges(in_uv, excluded)
    corrected = interpolate_motion(
        EstimationClockRecording(masked, clock),
        motion,
        border_mode=resolved_interpolation["border_mode"],
        spatial_interpolation_method=resolved_interpolation[
            "spatial_interpolation_method"
        ],
        sigma_um=float(resolved_interpolation["sigma_um"]),
        p=int(resolved_interpolation["p"]),
        num_closest=int(resolved_interpolation["num_closest"]),
        interpolation_time_bin_centers_s=None,
        interpolation_time_bin_edges_s=None,
        interpolation_time_bin_size_s=None,
        # Stated rather than left to ``dtype=None``, which inherits the input
        # dtype and raises for an integer one (``sortingcomponents/motion/
        # motion_interpolation.py:397-401``).
        dtype=in_uv.get_dtype(),
    )
    kept = set(corrected.channel_ids.tolist())
    removed = [c for c in recording.channel_ids.tolist() if c not in kept]
    if corrected.get_num_channels() == 0:
        raise ValueError(
            "Motion correction: border_mode='remove_channels' removed every "
            f"channel ({removed}); the estimated displacement moves each "
            "contact outside the probe in some temporal bin. Use "
            "'force_extrapolate' or inspect the estimate."
        )
    positions = np.asarray(corrected.get_channel_locations(), dtype=float)
    if not np.isfinite(positions).all() or len(
        np.unique(np.round(positions, 6), axis=0)
    ) != len(positions):
        raise ValueError(
            "Motion correction: the corrected recording's contact positions "
            f"must be finite and distinct; got {positions.tolist()}."
        )
    return AppliedMotion(
        recording=silence_frame_ranges(corrected, excluded),
        removed_channel_ids=removed,
    )


def motion_corrected_identity_payload(
    *,
    motion_estimate_id,
    motion_interpolation_params_name: str,
    resolved_params_hash: str,
    spikeinterface_version: str,
    motion_interpolation_algorithm_version: int,
) -> dict:
    """The logical identity a ``motion_corrected_recording_id`` is derived from.

    Parameters
    ----------
    motion_estimate_id : uuid.UUID or str
        The saved estimate (it carries the source, mask, estimation recipe,
        SpikeInterface version and estimation algorithm version).
    motion_interpolation_params_name : str
    resolved_params_hash : str
        :func:`resolved_params_hash` of the resolved interpolation.
    spikeinterface_version : str
        SpikeInterface whose ``interpolate_motion`` applies the estimate.
    motion_interpolation_algorithm_version : int

    Returns
    -------
    dict
        Payload for ``_selection_identity.deterministic_id``.
    """
    return {
        "motion_estimate_id": motion_estimate_id,
        "motion_interpolation_params_name": motion_interpolation_params_name,
        "resolved_params_hash": resolved_params_hash,
        "spikeinterface_version": spikeinterface_version,
        "motion_interpolation_algorithm_version": int(
            motion_interpolation_algorithm_version
        ),
    }


def motion_corrected_recording_artifact_lock(
    motion_corrected_recording_id, *, timeout: float = -1
):
    """Return a cross-process lock serializing one corrected artifact's slot.

    The analog of :func:`._recording_fingerprint.recording_artifact_lock`
    for a motion-corrected recording: a rebuild of the same artifact never
    interleaves with another. Its filename prefix keeps it apart from the
    recording and concat locks.

    Parameters
    ----------
    motion_corrected_recording_id
        The corrected recording whose canonical artifact the caller mutates.
    timeout : float, optional
        Seconds to wait before raising ``filelock.Timeout``; ``-1`` (default)
        blocks.

    Returns
    -------
    filelock.FileLock
        An unacquired lock.
    """
    from filelock import FileLock

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_cache_root

    root = analyzer_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(
        str(
            root
            / f"motion_corrected_{motion_corrected_recording_id}.artifact.lock"
        ),
        timeout=timeout,
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


class MotionSelectionIdentity(NamedTuple):
    """The derived id of a motion selection and the master row it names.

    Attributes
    ----------
    selection_id : uuid.UUID
        The deterministic ``motion_estimate_id`` or
        ``motion_corrected_recording_id``.
    master_row : dict
        The selection master's secondary attributes the id was derived from.
    """

    selection_id: object
    master_row: dict


def motion_estimate_selection_identity(
    *,
    source_kind: str,
    source_id,
    source_content_hash: str,
    artifact_detection_id,
    motion_estimation_params_name: str,
    estimation_params: dict,
) -> MotionSelectionIdentity:
    """Derive a ``MotionEstimateSelection`` id without touching the database.

    Resolves the recipe blob against the installed SpikeInterface and stamps
    the SpikeInterface and algorithm versions, exactly as
    ``MotionEstimateSelection.insert_selection`` does, so a planner can
    preview the id the insert will mint.

    Parameters
    ----------
    source_kind : {"recording", "concatenated_recording"}
    source_id : uuid.UUID or str
        The ``recording_id`` or ``concat_recording_id``.
    source_content_hash : str
        The source artifact's persisted ``content_hash``.
    artifact_detection_id : uuid.UUID, str or None
    motion_estimation_params_name : str
    estimation_params : dict
        That ``MotionEstimationParameters`` row's ``params`` blob.

    Returns
    -------
    MotionSelectionIdentity
    """
    import spikeinterface

    from spyglass.spikesorting.v2._selection_identity import deterministic_id

    master_row = {
        "motion_estimation_params_name": motion_estimation_params_name,
        "resolved_params_hash": resolved_params_hash(
            resolve_estimation_params(estimation_params)
        ),
        "spikeinterface_version": spikeinterface.__version__,
        "motion_algorithm_version": MOTION_ALGORITHM_VERSION,
        "source_content_hash": str(source_content_hash),
    }
    selection_id = deterministic_id(
        "motion_estimate",
        motion_estimate_identity_payload(
            source_kind=source_kind,
            source_id=source_id,
            artifact_detection_id=artifact_detection_id,
            **master_row,
        ),
    )
    return MotionSelectionIdentity(selection_id, master_row)


def motion_corrected_selection_identity(
    *,
    motion_estimate_id,
    motion_interpolation_params_name: str,
    interpolation_params: dict,
) -> MotionSelectionIdentity:
    """Derive a ``MotionCorrectedRecordingSelection`` id without the database.

    The same derivation ``MotionCorrectedRecordingSelection.insert_selection``
    uses; the estimate need not be populated for the id to be known.

    Parameters
    ----------
    motion_estimate_id : uuid.UUID or str
    motion_interpolation_params_name : str
    interpolation_params : dict
        That ``MotionInterpolationParameters`` row's ``params`` blob.

    Returns
    -------
    MotionSelectionIdentity
    """
    import uuid

    import spikeinterface

    from spyglass.spikesorting.v2._selection_identity import deterministic_id

    master_row = {
        "motion_estimate_id": uuid.UUID(str(motion_estimate_id)),
        "motion_interpolation_params_name": motion_interpolation_params_name,
        "resolved_params_hash": resolved_params_hash(
            resolve_interpolation_params(interpolation_params)
        ),
        "spikeinterface_version": spikeinterface.__version__,
        "motion_interpolation_algorithm_version": (
            MOTION_INTERPOLATION_ALGORITHM_VERSION
        ),
    }
    selection_id = deterministic_id(
        "motion_corrected_recording",
        motion_corrected_identity_payload(**master_row),
    )
    return MotionSelectionIdentity(selection_id, master_row)


def motion_input_fingerprint(
    *,
    source_content_hash: str,
    artifact_detection_id,
    n_samples: int,
    sampling_frequency: float,
    continuity_spans,
    continuity_start_s,
    continuity_end_s,
    statistics_spans,
    channel_ids,
    channel_locations,
    resolved_params_hash: str,
) -> str:
    """SHA-256 of everything an estimate was computed from.

    The source content, mask choice, frame spans and the continuity spans'
    first and last timestamps, estimation channels and their positions, and
    the resolved configuration (which holds the gap cap). Two estimates with
    the same fingerprint read the same valid samples on the same estimation
    clock and geometry with the same settings.

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
        "continuity_start_s": np.asarray(
            continuity_start_s, dtype=np.float64
        ).tolist(),
        "continuity_end_s": np.asarray(
            continuity_end_s, dtype=np.float64
        ).tolist(),
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
