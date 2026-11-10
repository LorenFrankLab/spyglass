"""Canonical manual exclusions and sample-exact subtraction from valid time."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from spyglass.spikesorting._intervals import merge_sorted_intervals
from spyglass.spikesorting.v2._core.lookup_validation import lossless_int
from spyglass.spikesorting.v2._core.signal_math import subtract_intervals


def resolve_manual_exclusions(value, *, concat=False):
    """Normalize runner input: intervals, or a concat member-index mapping."""
    if not concat:
        if isinstance(value, Mapping):
            raise TypeError(
                "Single-session manual_excluded_times must be [start, stop] "
                "intervals; a member_index mapping is for "
                "run_v2_concat_pipeline / estimate_concat_motion."
            )
        return normalize_manual_exclusions(value)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "Concat manual_excluded_times must map member_index to intervals."
        )

    result = {}
    for index, times in value.items():
        intervals = normalize_manual_exclusions(times)
        if intervals:
            result[lossless_int(index, "member_index")] = intervals
    return result


def artifact_recipe_with_manual_exclusions(bundle, exclusions):
    """Manual masks still need a detection output when automatic scanning is off."""
    if exclusions and bundle.artifact_detection_params_name is None:
        return bundle.model_copy(
            update={"artifact_detection_params_name": "none"}
        )
    return bundle


def normalize_manual_exclusions(intervals):
    """Sorted union of finite [start, stop) intervals in session seconds."""
    if intervals is None:
        return []
    values = np.asarray(intervals, dtype=float)
    if values.size == 0:
        return []
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or not np.isfinite(values).all()
        or (values[:, 1] <= values[:, 0]).any()
    ):
        raise ValueError(
            "manual_excluded_times must contain finite [start, stop) pairs "
            "with start < stop, in original session seconds."
        )
    return merge_sorted_intervals(sorted(values.tolist()))


def apply_manual_exclusions(valid_times, excluded_times, min_length_s):
    """Subtract exclusions without allocating a timestamp or per-sample mask.

    Input exclusions are half-open; valid intervals preserve the pipeline's
    inclusive recording endpoints and internal cut boundaries. Work on each
    already-valid interval so no output spans a recording gap or restores a
    detected artifact.
    """
    exclusions = normalize_manual_exclusions(excluded_times)
    if not exclusions or len(valid_times) == 0:
        return valid_times
    # ``inclusive_stop`` ends a piece cut at a valid stop on the predecessor
    # float, so an excluded final sample differs from an inclusive endpoint.
    kept = subtract_intervals(valid_times, exclusions, inclusive_stop=True)
    # Untouched intervals were already accepted upstream; only cut pieces drop.
    untouched = {(float(start), float(stop)) for start, stop in valid_times}
    times = np.asarray(kept, dtype=float).reshape(-1, 2)
    keep = [
        stop - start >= min_length_s or (start, stop) in untouched
        for start, stop in kept
    ]
    return times[np.asarray(keep, dtype=bool)]
