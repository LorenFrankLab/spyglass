"""Canonicalize v1 and v2 sort inputs for the v1/v2 parity fingerprint.

Used by
:func:`tests.spikesorting.v2.single_session.test_ground_truth_parity.test_v2_real_data_v1_parity`
so an input-layer divergence surfaces before the spike-time comparison.

JSON serialization rules:

* numpy scalars cast to Python scalars;
* numpy arrays serialize as
  ``{"_array_data": list, "_array_meta": {"dtype": str, "shape": list}}``;
* tuples normalize to lists;
* int-keyed dicts (e.g. ``bad_channel_by_electrode_id``) are rebuilt by
  the caller, which owns which dicts are int-keyed.
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

    See the module docstring for the rules; other scalars pass through.
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

    Floats compare via :func:`math.isclose` (``rel_tol=1e-9``) to absorb
    JSON round-trip drift. Mismatches raise :class:`AssertionError` naming
    the path, e.g. ``root.electrode_ids[3]``. Symmetric in ``left`` /
    ``right``.

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
        if not (
            isinstance(left, (int, float)) and isinstance(right, (int, float))
        ):
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


# Keys that never reach the SI sorter call; dropped on both sides.
_SCHEMA_ONLY_SORTER_KEYS: frozenset[str] = frozenset({"schema_version"})

_LEGACY_SORTER_KEYS: dict[str, frozenset[str]] = {
    "clusterless_thresholder": frozenset({"outputs", "random_chunk_kwargs"}),
    "mountainsort4": frozenset({"outputs"}),
}


def canonical_sorter(sorter: str, params: dict) -> dict:
    """Return the canonical fingerprint dict for a sorter's params.

    Drops keys SI never receives, then normalizes. For
    ``clusterless_thresholder``, ``noise_levels=None`` (v2 default) and
    an omitted key (v1 row) both mean SI auto-estimates per-channel MAD,
    so they canonicalize identically. Unknown sorters are only normalized.
    """
    drop = (
        _LEGACY_SORTER_KEYS.get(sorter, frozenset()) | _SCHEMA_ONLY_SORTER_KEYS
    )
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

    v1 rows are flat (``frequency_min`` / ``frequency_max`` /
    ``min_segment_length``); v2 rows are nested
    (``bandpass_filter`` / ``common_reference``). The canonical form is
    the four fields that reach SI:

    * ``freq_min``, ``freq_max`` (Hz),
    * ``reference_operator`` (``"median"`` or ``"average"``),
    * ``min_segment_length`` (s).

    Dropped: v1 ``seed`` / ``margin_ms`` (v2 hardcodes both) and v2
    ``schema_version`` / ``whiten`` (v2 whitens in the sorter).
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


# v1 concurrency knobs; v2 keeps them in ``job_kwargs``. No scientific effect.
_V1_JOB_KWARG_KEYS = frozenset({"chunk_duration", "n_jobs", "progress_bar"})


def _derive_detect(params: dict) -> bool:
    """Return effective ``detect`` from v1's "absent ⇒ inferred" semantics.

    v1 has no ``detect`` field; detection is off when both thresholds
    are ``None``.
    """
    if "detect" in params:
        return bool(params["detect"])
    amp = _artifact_param(
        params, "amplitude_threshold_uv", "amplitude_thresh_uV"
    )
    zsc = _artifact_param(params, "zscore_threshold", "zscore_thresh")
    return not (amp is None and zsc is None)


def _artifact_param(params: dict, new_name: str, v1_name: str, default=None):
    """Read a v2 artifact parameter, accepting the historical v1 name."""
    if new_name in params:
        return params[new_name]
    if v1_name in params:
        return params[v1_name]
    return default


def canonical_artifact(params: dict) -> dict:
    """Extract the SI-effective artifact-detection kwargs from v1 or v2 params.

    Strips v1 concurrency knobs and v2's ``schema_version``, then fills
    the v2 schema defaults (``join_window_ms=1.0``,
    ``min_length_s=1.0``, ``proportion_above_threshold=1.0``,
    ``removal_window_ms=1.0``) when v1's row omits the field. The
    canonical form is the eight scientific fields that reach SI:

    * ``detect`` (derived from threshold presence on v1, explicit on v2),
    * ``amplitude_threshold_uv`` / v1 ``amplitude_thresh_uV`` and
      ``zscore_threshold`` / v1 ``zscore_thresh`` (at least one non-None
      unless ``detect=False``),
    * ``proportion_above_threshold``, ``removal_window_ms``,
    * ``join_window_ms``, ``min_length_s`` (v2-only fields; v1
      rows are read with the v2 defaults applied).

    Filling v2 defaults on the v1 side makes default rows fingerprint
    identically; an explicit v1 value that differs is a mismatch.
    """
    p = _normalize(params)
    amplitude_threshold_uv = _artifact_param(
        p, "amplitude_threshold_uv", "amplitude_thresh_uV"
    )
    zscore_threshold = _artifact_param(p, "zscore_threshold", "zscore_thresh")
    proportion_above_threshold = _artifact_param(
        p,
        "proportion_above_threshold",
        "proportion_above_thresh",
        default=1.0,
    )
    return {
        "detect": _derive_detect(p),
        "amplitude_threshold_uv": (
            float(amplitude_threshold_uv)
            if amplitude_threshold_uv is not None
            else None
        ),
        "zscore_threshold": (
            float(zscore_threshold) if zscore_threshold is not None else None
        ),
        "proportion_above_threshold": float(proportion_above_threshold),
        "removal_window_ms": float(p.get("removal_window_ms", 1.0)),
        "join_window_ms": float(p.get("join_window_ms", 1.0)),
        "min_length_s": float(p.get("min_length_s", 1.0)),
    }
