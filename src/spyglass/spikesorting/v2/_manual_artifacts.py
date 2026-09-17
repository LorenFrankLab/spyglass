"""Canonical manual exclusions and sample-exact subtraction from valid time."""

from __future__ import annotations


def resolve_manual_exclusions(value, *, concat=False):
    """Normalize runner input: intervals, or a concat member-index mapping."""
    from collections.abc import Mapping

    if not concat:
        return normalize_manual_exclusions(value)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "Concat manual_excluded_times must map member_index to intervals."
        )
    from spyglass.spikesorting.v2._lookup_validation import lossless_int

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
    import numpy as np

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
    merged = []
    for start, stop in sorted(values.tolist()):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(stop, merged[-1][1])
        else:
            merged.append([start, stop])
    return merged


def apply_manual_exclusions(valid_times, excluded_times, min_length_s):
    """Subtract exclusions without allocating a timestamp or per-sample mask.

    Input exclusions are half-open; valid intervals preserve the pipeline's
    inclusive recording endpoints and internal cut boundaries. Work on each
    already-valid interval so no output spans a recording gap or restores a
    detected artifact.
    """
    import numpy as np

    exclusions = normalize_manual_exclusions(excluded_times)
    if not exclusions or len(valid_times) == 0:
        return valid_times
    kept = []
    for start, stop in valid_times:
        cursor = float(start)
        for bad_start, bad_stop in exclusions:
            if bad_stop <= cursor:
                continue
            if bad_start > stop:
                break
            if cursor < bad_start:
                # Internal valid stops map to the first excluded sample. The
                # predecessor float also distinguishes a deliberately excluded
                # final sample from an inclusive recording/chunk endpoint.
                cut = (
                    np.nextafter(bad_start, -np.inf)
                    if bad_start == stop
                    else bad_start
                )
                kept.append((cursor, cut))
            cursor = max(cursor, bad_stop)
        if cursor <= stop:
            kept.append((cursor, float(stop)))
    times = np.asarray(kept, dtype=float).reshape(-1, 2)
    return times[(times[:, 1] - times[:, 0]) >= min_length_s]
