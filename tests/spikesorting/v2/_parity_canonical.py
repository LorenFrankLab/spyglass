"""Canonicalization helpers for v1↔v2 parity invariant fingerprints.

The v1↔v2 parity gate at
:func:`tests.spikesorting.v2.test_single_session_pipeline.test_v2_real_data_v1_parity`
fingerprints the inputs to the sort (NWB hash, sort-group electrodes,
preprocessing params, artifact params, sorter params, valid-times
hash) so a divergence at the input layer surfaces *before* the
spike-time output comparison runs. The helpers in this module
normalize either v1's flat-dict / legacy-keys schema or v2's
Pydantic-validated nested schema into a single canonical dict so the
fingerprint check is a pure value comparison.

JSON serialization rules (see ``parity-extensions.md`` §
"Serialization rules"):

* numpy scalars cast to Python scalars before serialization;
* numpy arrays serialize as
  ``{"_array_data": list, "_array_meta": {"dtype": str, "shape": list}}``
  so dtype + shape survive JSON;
* tuples normalize to lists (JSON has no tuple);
* int-keyed dicts (e.g. ``bad_channel_by_electrode_id``) round-trip
  through stringified keys and are reconstructed on read -- handled
  by the caller, NOT here, since the choice of which dicts are
  int-keyed is a caller-side contract.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Internal normalization
# ---------------------------------------------------------------------------


def _normalize(value: Any) -> Any:
    """Recursively normalize a parameter value for JSON-stable comparison.

    Numpy scalars cast to Python scalars, numpy arrays serialize with
    explicit dtype + shape metadata, tuples flatten to lists, and
    dicts / lists recurse. Plain Python scalars, strings, ``None``,
    and booleans pass through unchanged.
    """
    if isinstance(value, np.ndarray):
        return {
            "_array_data": value.tolist(),
            "_array_meta": {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            },
        }
    if isinstance(value, np.generic):  # numpy scalar (int64, float64, ...)
        return value.item()
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Recursive equality with float tolerance + pretty path
# ---------------------------------------------------------------------------


_REL_TOL = 1e-9
_ABS_TOL = 0.0


def _format_value(value: Any) -> str:
    """Compact stringification for assertion messages."""
    if isinstance(value, float):
        return repr(value)
    return repr(value)


def assert_canonical_dict_equal(
    left: Any, right: Any, *, path: str = "root"
) -> None:
    """Assert two canonical structures are equal with a path-tagged diff.

    Floats compare via :func:`math.isclose` (``rel_tol=1e-9``,
    ``abs_tol=0.0``) so JSON round-trip precision drift does not
    trip the fingerprint check. Type mismatches, missing keys, and
    elementwise list disagreements raise :class:`AssertionError`
    with a path like ``root.canonical_preproc.freq_min`` or
    ``root.electrode_ids[3]`` so the failing field is obvious.

    The helper is symmetric in ``left`` / ``right`` -- if either
    side has a key the other lacks, the missing-key path is named
    in the message.

    Parameters
    ----------
    left, right
        The two canonical structures (dicts, lists, scalars,
        ``None``, ``bool``) to compare.
    path
        Caller-side label for the root; the helper prepends it to
        each nested key so the assertion message stays grep-able.
    """
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            raise AssertionError(
                f"{path}: type mismatch -- left={type(left).__name__} "
                f"right={type(right).__name__}"
            )
        left_keys = set(left)
        right_keys = set(right)
        only_left = left_keys - right_keys
        only_right = right_keys - left_keys
        if only_left or only_right:
            parts = []
            if only_left:
                parts.append(f"left-only={sorted(only_left)}")
            if only_right:
                parts.append(f"right-only={sorted(only_right)}")
            raise AssertionError(f"{path}: key mismatch -- {'; '.join(parts)}")
        for key in sorted(left):
            assert_canonical_dict_equal(
                left[key], right[key], path=f"{path}.{key}"
            )
        return

    if isinstance(left, list) or isinstance(right, list):
        if not (isinstance(left, list) and isinstance(right, list)):
            raise AssertionError(
                f"{path}: type mismatch -- left={type(left).__name__} "
                f"right={type(right).__name__}"
            )
        if len(left) != len(right):
            raise AssertionError(
                f"{path}: list length mismatch -- "
                f"left={len(left)} right={len(right)}"
            )
        for idx, (lv, rv) in enumerate(zip(left, right)):
            assert_canonical_dict_equal(lv, rv, path=f"{path}[{idx}]")
        return

    # Booleans are a subclass of int in Python; check explicitly so
    # ``True == 1`` does not silently pass a parity check.
    if isinstance(left, bool) or isinstance(right, bool):
        if type(left) is not type(right) or left != right:
            raise AssertionError(
                f"{path}: bool mismatch -- left={_format_value(left)} "
                f"right={_format_value(right)}"
            )
        return

    if left is None or right is None:
        if left is not right:
            raise AssertionError(
                f"{path}: None mismatch -- left={_format_value(left)} "
                f"right={_format_value(right)}"
            )
        return

    if isinstance(left, float) or isinstance(right, float):
        # Mixed int/float compare numerically via isclose; pure int/int
        # falls through to the strict equality branch below.
        if not (isinstance(left, (int, float)) and isinstance(right, (int, float))):
            raise AssertionError(
                f"{path}: type mismatch -- left={type(left).__name__} "
                f"right={type(right).__name__}"
            )
        if not math.isclose(
            float(left), float(right), rel_tol=_REL_TOL, abs_tol=_ABS_TOL
        ):
            raise AssertionError(
                f"{path}: float mismatch -- left={_format_value(left)} "
                f"right={_format_value(right)}"
            )
        return

    if type(left) is not type(right):
        raise AssertionError(
            f"{path}: type mismatch -- left={type(left).__name__} "
            f"right={type(right).__name__}"
        )
    if left != right:
        raise AssertionError(
            f"{path}: value mismatch -- left={_format_value(left)} "
            f"right={_format_value(right)}"
        )


# ---------------------------------------------------------------------------
# canonical_sorter
# ---------------------------------------------------------------------------


# Schema-only keys that v1's ``SpikeSorterParameters`` blob carries
# but never reach the SI sorter call. Dropping them on both sides
# keeps the fingerprint stable across the v1 / v2 schema asymmetry.
#
# ``schema_version`` is stamped by v2's Pydantic schema and absent
# from v1; strip on both sides regardless of sorter.
_SCHEMA_ONLY_SORTER_KEYS: frozenset[str] = frozenset({"schema_version"})

_LEGACY_SORTER_KEYS: dict[str, frozenset[str]] = {
    "clusterless_thresholder": frozenset({"outputs", "random_chunk_kwargs"}),
    "mountainsort4": frozenset({"outputs"}),
}


def canonical_sorter(sorter: str, params: dict) -> dict:
    """Return the canonical fingerprint dict for a sorter's params.

    Strips schema-only / legacy keys that v1 stamps but SI never
    receives, then normalizes the remaining values. For
    ``clusterless_thresholder``, ``noise_levels=None`` is treated as
    "field absent" so an explicit ``None`` (v2 Pydantic default) and
    an omitted key (v1 production row) canonicalize identically --
    both mean "let SI auto-estimate per-channel MAD."

    Unknown sorters get pure normalization with no key drops, so a
    future sorter added to the parity matrix degrades gracefully
    until the canonical rules are extended.
    """
    drop = _LEGACY_SORTER_KEYS.get(sorter, frozenset()) | _SCHEMA_ONLY_SORTER_KEYS
    cleaned = {k: v for k, v in params.items() if k not in drop}
    if sorter == "clusterless_thresholder":
        if cleaned.get("noise_levels", "__sentinel__") is None:
            cleaned.pop("noise_levels")
    return _normalize(cleaned)


# ---------------------------------------------------------------------------
# canonical_preproc
# ---------------------------------------------------------------------------


def canonical_preproc(params: dict) -> dict:
    """Extract the SI-effective preprocessing kwargs from v1 or v2 params.

    v1's ``SpikeSortingPreprocessingParameters`` row is a flat dict
    keyed ``frequency_min`` / ``frequency_max`` / ``min_segment_length``;
    v2's :class:`PreprocessingParamsSchema` is a nested Pydantic dump
    with ``bandpass_filter`` / ``common_reference`` sub-blocks plus a
    ``schema_version`` stamp. Both encode an effective
    "bandpass + common-reference + min-segment" preprocessing
    pipeline; the canonical form is the four fields that actually
    reach SI:

    * ``freq_min``, ``freq_max`` (Hz),
    * ``reference_operator`` (``"median"`` or ``"average"``),
    * ``min_segment_length`` (s).

    v1's ``seed`` and ``margin_ms`` are dropped (v2 hardcodes both
    in the runtime path, so they are schema-only on v1). v2's
    ``schema_version`` and ``whiten`` block are dropped (v2 defers
    whitening to the sorter; the field is forward-compat
    scaffolding, see ``v2/_params/preprocessing.py``).
    """
    p = _normalize(params)
    if "bandpass_filter" in p or "schema_version" in p:
        bp = p.get("bandpass_filter", {})
        ref = p.get("common_reference", {})
        return {
            "freq_min": float(bp.get("freq_min", 300.0)),
            "freq_max": float(bp.get("freq_max", 6000.0)),
            "reference_operator": ref.get("operator", "median"),
            "min_segment_length": float(p.get("min_segment_length", 1.0)),
        }
    # v1 flat form.
    return {
        "freq_min": float(p.get("frequency_min", 300.0)),
        "freq_max": float(p.get("frequency_max", 6000.0)),
        "reference_operator": p.get("reference_operator", "median"),
        "min_segment_length": float(p.get("min_segment_length", 1.0)),
    }


# ---------------------------------------------------------------------------
# canonical_artifact
# ---------------------------------------------------------------------------


# v1's artifact-params blob carried concurrency knobs that v2
# factored out into a separate ``job_kwargs`` row; they never affect
# the scientific output of artifact detection.
_V1_JOB_KWARG_KEYS = frozenset({"chunk_duration", "n_jobs", "progress_bar"})


def _derive_detect(params: dict) -> bool:
    """Return effective ``detect`` from v1's "absent ⇒ inferred" semantics.

    v1 had no ``detect`` field; the runtime path inspected whether
    both threshold fields were ``None`` (the ``"none"`` preset) and
    disabled detection in that case. v2 promoted the inference to
    an explicit boolean.
    """
    if "detect" in params:
        return bool(params["detect"])
    amp = params.get("amplitude_thresh_uV")
    zsc = params.get("zscore_thresh")
    return not (amp is None and zsc is None)


def canonical_artifact(params: dict) -> dict:
    """Extract the SI-effective artifact-detection kwargs from v1 or v2 params.

    Strips v1's runtime concurrency knobs (``chunk_duration``,
    ``n_jobs``, ``progress_bar``) and v2's ``schema_version`` stamp,
    then fills the v2 schema defaults (``join_window_ms=1.0``,
    ``min_length_s=1.0``, ``proportion_above_thresh=1.0``,
    ``removal_window_ms=1.0``) when v1's row omits the field. The
    canonical form is the eight scientific fields that reach SI:

    * ``detect`` (derived from threshold presence on v1, explicit on v2),
    * ``amplitude_thresh_uV``, ``zscore_thresh`` (at least one
      non-None unless ``detect=False``),
    * ``proportion_above_thresh``, ``removal_window_ms``,
    * ``join_window_ms``, ``min_length_s`` (v2-only fields; v1
      rows are read with the v2 defaults applied).

    Filling v2 defaults on the v1 side is the "v1's absent fields
    silently followed the v2 schema defaults" reading -- it is the
    only reading under which a v1 default row and a v2 default row
    fingerprint identically. A v1 row with an explicit field
    diverging from the v2 default surfaces as a canonical mismatch.
    """
    p = _normalize(params)
    return {
        "detect": _derive_detect(p),
        "amplitude_thresh_uV": (
            float(p["amplitude_thresh_uV"])
            if p.get("amplitude_thresh_uV") is not None
            else None
        ),
        "zscore_thresh": (
            float(p["zscore_thresh"])
            if p.get("zscore_thresh") is not None
            else None
        ),
        "proportion_above_thresh": float(p.get("proportion_above_thresh", 1.0)),
        "removal_window_ms": float(p.get("removal_window_ms", 1.0)),
        "join_window_ms": float(p.get("join_window_ms", 1.0)),
        "min_length_s": float(p.get("min_length_s", 1.0)),
    }
