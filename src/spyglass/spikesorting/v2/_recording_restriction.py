"""Time/channel restriction + truncation-guard arithmetic behind ``Recording``.

These functions slice the raw recording in time and channels for
``Recording.make_compute`` (and the rebuild path), and resolve the
intended-vs-requested save durations and the sample-grid tolerance that the
``make_insert`` truncation / over-request guards compare against. The table
threads already-fetched DB state in (the tri-part
``make_fetch``/``make_compute``/``make_insert`` contract forbids DB I/O inside
compute), so the restriction hot path here is DB-free.

Why this lives in its own module rather than in ``recording.py``:
``recording.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part dependencies. The restriction logic
needs none of that at import, so ``Recording`` becomes a thin orchestrator
(fetch -> call these -> insert / verify). Same "thin DataJoint shell over
pure/IO services" direction as ``_artifact_compute`` / ``_selection_identity``
/ ``_analyzer_cache`` / ``_curation_transforms`` / ``_units_nwb`` /
``_sorting_compute``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all SpikeInterface / numpy / spyglass dependencies are
imported lazily inside the functions. ``restrict_recording`` touches the DB at
CALL time via its lazy ``Interval`` import (which opens a DB connection on
import) and the lazy ``spikeinterface_channel_ids`` it calls (an ``Nwbfile``
path resolution in ``_recording_geometry``). The duration arithmetic
(``truncation_tolerance``, ``compute_recording_save_expectation``) is pure.
"""

from __future__ import annotations

from typing import NamedTuple


def truncation_tolerance(
    n_intended_intervals: int, sampling_frequency: float
) -> float:
    """Compute the sample-grid tolerance for the truncation guard.

    ``expected_saved_total`` is a continuous interval-length sum while
    the saved duration is the sample-snapped span of the time-stripped
    concat. Each consolidated interval's ``[start, end]`` is snapped to
    the raw sample grid INDEPENDENTLY, so the per-boundary quantization
    error (up to ~1 sample per interval) accumulates with the interval
    count; the ``+1.5`` then covers the ``(N-1)/fs`` concat off-by-one.

    A fixed ``1.5 / fs`` slack (the pre-fix value) false-positives on
    legitimate disjoint multi-epoch sorts -- numerically ~13% of
    single-interval and ~40% of 20-interval requests -- and then
    deletes the just-written file. Scaling by interval count drops that
    to 0% while still catching genuine packet loss / interval
    misalignment, which drops far more than ``n + 1.5`` samples.

    Parameters
    ----------
    n_intended_intervals : int
        Number of consolidated intervals that were saved; the tolerance
        scales with this count to cover per-boundary quantization error.
    sampling_frequency : float
        Recording sampling frequency in Hz.

    Returns
    -------
    float
        Tolerance in seconds for the ``make_insert`` truncation guard.
    """
    return (n_intended_intervals + 1.5) / sampling_frequency


class RecordingSaveExpectation(NamedTuple):
    """Intended-vs-requested save durations for the truncation guards.

    Attributes
    ----------
    expected_saved_total : float
        Seconds the sort intervals cover AFTER intersecting with the raw valid
        times and dropping sub-``min_segment_length`` slivers -- the same
        filtering ``restrict_recording`` applies, so it matches what is actually
        written. The ``make_insert`` truncation guard compares the saved
        duration against this (not the raw request) so intentionally-dropped
        slivers are not mis-flagged as truncation.
    n_intended_intervals : int
        Number of consolidated intended intervals; the truncation tolerance
        scales with this (per-boundary sample-grid error accumulates).
    requested_saved_total : float
        Seconds the sort intervals request AFTER ``min_segment_length``
        filtering but WITHOUT clipping to raw coverage.
    over_request : float
        ``requested_saved_total - expected_saved_total``: the span requested
        past the raw recording's coverage. A positive value is the clip
        ``make_compute`` surfaces as a warning.
    """

    expected_saved_total: float
    n_intended_intervals: int
    requested_saved_total: float
    over_request: float


def compute_recording_save_expectation(
    intended_intervals, sort_valid_times, min_segment_length: float
) -> RecordingSaveExpectation:
    """Resolve the intended/requested save durations for a sort interval.

    Pure (DB-free) duration arithmetic for the ``make_insert`` truncation and
    over-request guards. The caller passes ``intended_intervals`` -- the result
    of ``Interval(sort_valid_times).intersect(Interval(raw_valid_times),
    min_length=min_segment_length).times``, the SAME filtering
    ``restrict_recording`` applies to build the recording -- so ``Interval``
    (which opens a DB connection on import) stays at the call site and this
    helper has no DB dependency.

    ``expected_saved_total`` sums those intended intervals; ``requested_saved_
    total`` sums the raw request after ``min_segment_length`` filtering but
    WITHOUT the raw clip. A positive ``over_request`` is exactly the span a sort
    interval asked for past the raw recording's coverage. Inter-segment gaps and
    sub-``min_segment_length`` slivers cancel out (excluded from both sides), so
    ``over_request`` fires only on a genuine request-past-coverage, not on
    intentional drops.

    Parameters
    ----------
    intended_intervals : array-like, shape (k, 2)
        Sort intervals already intersected with the raw valid times and
        min_segment_length-filtered (what will actually be saved).
    sort_valid_times : array-like, shape (n, 2)
        The originally requested sort intervals ``[start, end]`` in seconds,
        before the raw-coverage clip.
    min_segment_length : float
        Minimum interval length (seconds); shorter requested slivers are
        excluded from ``requested_saved_total`` (mirroring the intersect's drop).

    Returns
    -------
    RecordingSaveExpectation
    """
    expected_saved_total = float(
        sum(end - start for start, end in intended_intervals)
    )
    requested_saved_total = float(
        sum(
            end - start
            for start, end in sort_valid_times
            if (end - start) >= min_segment_length
        )
    )
    return RecordingSaveExpectation(
        expected_saved_total=expected_saved_total,
        n_intended_intervals=len(intended_intervals),
        requested_saved_total=requested_saved_total,
        over_request=requested_saved_total - expected_saved_total,
    )


def restrict_recording(
    recording,
    nwb_file_name: str,
    interval_list_name: str,
    sort_group_channel_ids: list,
    reference_mode: str,
    reference_electrode_id: int | None,
    sort_valid_times,
    raw_valid_times,
    *,
    min_segment_length: float = 1.0,
    bad_channel_handling: str = "remove",
    bad_channel_ids=(),
):
    """Slice the SI recording in time and channels.

    Returns ``(recording, timestamps_override, n_selected_intervals)``:

    - ``recording`` is the time- and channel-restricted SI
      recording. A single selected interval yields a
      ``frame_slice``; several yield
      ``concatenate_recordings(sliced)``.
    - ``timestamps_override`` is the persisted wall-clock timestamp
      vector -- ``times[s:e]`` for the single-interval path, the
      concatenated per-interval slices for the multi-interval path.
      It carries the wall-clock gaps (for the concat) that SI's
      ``frame_slice`` /
      ``concatenate_recordings(ignore_times=True)`` drop from
      ``recording.get_times()``. The caller passes it to
      ``write_nwb_artifact`` as ``timestamps_override=``.
    - ``n_selected_intervals`` is the number of consolidated
      intervals (1 ==> single contiguous ``frame_slice``; >1 ==>
      gap-spanning concat). The caller derives the saved
      start/end/duration from the override only when this is 1, so
      the multi-interval gap envelope never feeds the truncation
      guard (which needs the gap-excluded span).

    The reference channel is included in the slice when it is
    a positive electrode id so ``common_reference`` can
    subtract it; it is dropped after referencing in
    ``apply_pre_motion_preprocessing``.

    Uses ``ChannelSliceRecording`` directly because SI 0.104
    dropped the ``recording.channel_slice(...)`` method that
    was available on v1's SI 0.99 -- the constructor accepts
    the same kwargs.

    Parameters
    ----------
    recording : si.BaseRecording
        The full-source SI recording to slice in time and channels.
    nwb_file_name : str
        Parent NWB filename, used to resolve SpikeInterface channel
        ids for the requested electrode ids.
    interval_list_name : str
        Name of the sort interval list; used only in the error
        message raised when the intersection is empty.
    sort_group_channel_ids : list
        Spyglass electrode ids of the sort group's declared members.
    reference_mode : str
        One of ``"none"``, ``"global_median"``, or ``"specific"``.
        On ``"specific"`` the reference electrode is sliced in for
        later subtraction.
    reference_electrode_id : int or None
        Electrode id of the ``"specific"`` reference; ignored for the
        other reference modes.
    sort_valid_times, raw_valid_times
        ``(n, 2)`` ndarrays of [start, end] seconds for the sort
        interval and the raw-data valid times. Fetched by
        ``make_fetch`` so the tri-part compute step does no DB
        I/O.
    min_segment_length : float, optional
        Drop intersected sub-intervals shorter than this many seconds
        before slicing. Default ``1.0``.
    bad_channel_handling : str, optional
        ``"remove"`` (default) or ``"interpolate"``. On
        ``"interpolate"`` the group's interior curated-bad channels
        are sliced in so they are present to be filled.
    bad_channel_ids : sequence of int, optional
        Interior curated-bad electrode ids to slice in for the
        ``"interpolate"`` path. Default ``()``.
    """
    import numpy as np
    from spikeinterface import concatenate_recordings
    from spikeinterface.core.channelslice import ChannelSliceRecording

    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting.v2._recording_geometry import (
        spikeinterface_channel_ids,
    )
    from spyglass.spikesorting.v2.utils import (
        _consolidate_intervals,
        _get_recording_timestamps,
        assert_reference_not_member,
    )

    # Route through ``_get_recording_timestamps`` so multi-segment
    # NWBs concatenate correctly. The single-segment path is
    # identical to ``recording.get_times()`` (delegation).
    times = _get_recording_timestamps(recording)

    # When the requested sort interval is disjoint (e.g., a
    # run+sleep+run epoch group), frame-slice each chunk
    # separately and concatenate; a single ``frame_slice`` on the
    # outer envelope silently sorts the inter-chunk gaps too.
    # Matches ``v1/recording.py:556-583``.
    intersection = Interval(sort_valid_times).intersect(
        Interval(raw_valid_times), min_length=min_segment_length
    )
    valid_times = intersection.times  # (n, 2) ndarray, seconds
    if len(valid_times) == 0:
        # After the min_segment_length filter the intersection may
        # be empty -- e.g. a noisy session where every chunk is
        # shorter than the threshold. Raise here instead of
        # crashing downstream on ``_consolidate_intervals`` index
        # access.
        raise ValueError(
            f"Recording.make: interval list {interval_list_name!r} "
            f"for {nwb_file_name!r} has zero intersection with raw "
            "data valid times after min_segment_length filtering "
            f"(min_segment_length={min_segment_length}s). Lower the "
            "threshold or fix the upstream IntervalList."
        )
    intervals_in_frames = _consolidate_intervals(valid_times, times)

    if len(intervals_in_frames) > 1:
        # ``concatenate_recordings`` defaults to ``ignore_times=True``,
        # which strips the wall-clock timestamps off each
        # ``frame_slice`` segment. Without explicit handling the
        # downstream ``write_nwb_artifact`` would call
        # ``recording.get_times()`` on the time-stripped concat
        # and get a synthetic 0-based array. Build the
        # concatenated-interval timestamps from ``times[s:e]``
        # slices (matching v1's
        # ``timestamps.extend(all_timestamps[start:end])``
        # pattern) and return them so the caller can pass the
        # array through to ``write_nwb_artifact`` as
        # ``timestamps_override``.
        sliced = [
            recording.frame_slice(start_frame=int(s), end_frame=int(e))
            for s, e in intervals_in_frames
        ]
        timestamps_override = np.concatenate(
            [times[int(s) : int(e)] for s, e in intervals_in_frames]
        )
        recording = concatenate_recordings(sliced)
    else:
        s, e = intervals_in_frames[0]
        recording = recording.frame_slice(start_frame=int(s), end_frame=int(e))
        # ``times`` is the full-source wall-clock vector; slice it
        # explicitly (rather than reading the frame-sliced
        # recording's ``get_times()``) so the persisted per-interval
        # timestamps match the multi-interval path above.
        timestamps_override = times[int(s) : int(e)]

    assert_reference_not_member(
        reference_mode, reference_electrode_id, sort_group_channel_ids
    )
    # Base slice = the sort group's declared members. Add the ``specific``
    # reference (sliced in only for subtraction, dropped after referencing) and,
    # on the ``interpolate`` path, the group's interior curated-bad channels so
    # they are present to be filled. ``remove`` adds neither curated-bad channel,
    # so the slice is byte-identical to today (the members, already sorted in
    # ``make_fetch``; ``sorted(set(...))`` preserves that order/dedup).
    extra: list[int] = []
    if reference_mode == "specific":
        extra.append(int(reference_electrode_id))
    if bad_channel_handling == "interpolate":
        extra.extend(int(c) for c in bad_channel_ids)
    slice_ids = sorted(set([int(c) for c in sort_group_channel_ids] + extra))

    si_ids = spikeinterface_channel_ids(nwb_file_name, slice_ids)
    recording = ChannelSliceRecording(
        recording,
        channel_ids=si_ids,
        renamed_channel_ids=slice_ids,
    )
    return recording, timestamps_override, len(intervals_in_frames)
