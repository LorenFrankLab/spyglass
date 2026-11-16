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


def ineffective_manual_exclusions(exclusions, valid_times) -> list:
    """Return the exclusions that overlap no valid interval.

    ``[a, b)`` overlaps an inclusive ``[s, e]`` iff ``a <= e`` and ``b > s``,
    the rule :func:`apply_manual_exclusions` cuts by, so an exclusion
    returned here would remove no sample.

    Parameters
    ----------
    exclusions : array-like
        ``(n_exclusions, 2)`` half-open ``[start, stop)`` session seconds.
    valid_times : array-like
        ``(n_intervals, 2)`` inclusive ``[start, stop]`` session seconds.

    Returns
    -------
    list of [float, float]
        The normalized exclusions that overlap no valid interval, sorted.
    """
    excluded = np.asarray(
        normalize_manual_exclusions(exclusions), dtype=float
    ).reshape(-1, 2)
    valid = np.asarray(valid_times, dtype=float).reshape(-1, 2)
    overlaps = (excluded[:, None, 0] <= valid[None, :, 1]) & (
        excluded[:, None, 1] > valid[None, :, 0]
    )  # (n_exclusions, n_intervals)
    return excluded[~overlaps.any(axis=1)].tolist()


def manual_exclusion_problem(
    exclusions,
    valid_times,
    *,
    nwb_file_name: str,
    interval_list_name: str,
    prefix: str = "",
) -> "str | None":
    """Explain exclusions that overlap none of the time a run sorts.

    Parameters
    ----------
    exclusions : array-like
        ``(n_exclusions, 2)`` half-open ``[start, stop)`` session seconds.
    valid_times : array-like
        ``(n_intervals, 2)`` time the recording keeps (the sort interval
        within the raw data valid times), in session seconds.
    nwb_file_name, interval_list_name : str
        The session and sort interval named in the message.
    prefix : str, optional
        Prepended to the message (e.g. the caller and concat member).

    Returns
    -------
    str or None
        The message, or ``None`` when every exclusion overlaps valid time.
    """
    bad = ineffective_manual_exclusions(exclusions, valid_times)
    if not bad:
        return None
    valid = np.asarray(valid_times, dtype=float).reshape(-1, 2)
    shown = ", ".join(f"[{a!r}, {b!r})" for a, b in bad[:5])
    if len(bad) > 5:
        shown += f", ... ({len(bad)} in all)"
    if len(valid):
        start = float(valid[0, 0])
        span = (
            f"{start:.3f} to {float(valid[-1, 1]):.3f} s in {len(valid)} "
            "interval(s), gaps excluded"
        )
        example = (
            " For example, 10 to 20 s after the start is "
            f"[[{start + 10.0!r}, {start + 20.0!r}]]."
        )
    else:
        span, example = "no time", ""
    return (
        f"{prefix}manual_excluded_times interval(s) {shown} overlap none of "
        f"the time this run sorts for {nwb_file_name!r} (interval_list_name "
        f"{interval_list_name!r} within 'raw data valid times'): {span}. "
        "Exclusions are absolute session timestamps (the IntervalList "
        "valid_times clock), not seconds from the recording start."
        f"{example} Correct or remove these interval(s); an interval inside "
        "an acquisition gap has no samples to exclude."
    )


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
