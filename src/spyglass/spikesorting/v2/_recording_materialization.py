"""Preprocessing / NWB-write services behind ``Recording``.

These functions are the materialization core of ``Recording.make_compute``
(and the rebuild path): slicing the raw recording in time and channels,
applying the pre-motion preprocessing stack (bandpass filter, then common
reference, no whitening), attaching legacy tetrode geometry, resolving
SpikeInterface
channel ids and per-channel probe metadata, building the
``ElectricalSeries.filtering`` provenance string, the truncation-guard
tolerance, and streaming the preprocessed traces into an
``AnalysisNwbfile`` (then hashing the persisted file for the cache
contract). The table threads already-fetched DB state in (the tri-part
``make_fetch``/``make_compute``/``make_insert`` contract forbids DB I/O
inside compute), so the preprocessing hot path here is DB-free.

Why this lives in its own module rather than in ``recording.py``:
``recording.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part dependencies. The preprocessing /
NWB-write logic needs none of that at import, so ``Recording`` becomes a
thin orchestrator (fetch -> call these -> insert / verify). Same "thin
DataJoint shell over pure/IO services" direction as ``_artifact_compute``
/ ``_selection_identity`` / ``_analyzer_cache`` / ``_curation_transforms``
/ ``_units_nwb`` / ``_sorting_compute``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all SpikeInterface / numpy / spyglass dependencies
are imported lazily inside the functions. Three functions inherently touch
the DB / DataJoint at CALL time via lazy imports: ``fetch_sort_group_probe_info``
(an ``Electrode * Probe`` fetch), ``spikeinterface_channel_ids`` (``Nwbfile``
path resolution), and ``write_nwb_artifact`` (``AnalysisNwbfile`` path
resolution + file create). ``write_nwb_artifact`` also lazily imports the
``_ELECTRICAL_SERIES_NAME`` constant from ``recording`` at call time -- by
then ``recording`` is fully imported, so there is no import cycle.
"""

from __future__ import annotations


def truncation_tolerance(
    n_intended_intervals: int, sampling_frequency: float
) -> float:
    """Sample-grid tolerance for the ``make_insert`` truncation guard.

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
    """
    return (n_intended_intervals + 1.5) / sampling_frequency


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
    sort_valid_times, raw_valid_times
        ``(n, 2)`` ndarrays of [start, end] seconds for the sort
        interval and the raw-data valid times. Fetched by
        ``make_fetch`` so the tri-part compute step does no DB
        I/O.
    """
    import numpy as np
    from spikeinterface import concatenate_recordings
    from spikeinterface.core.channelslice import ChannelSliceRecording

    from spyglass.common.common_interval import Interval
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


def spikeinterface_channel_ids(nwb_file_name: str, spyglass_ids):
    """Map Spyglass electrode_ids onto SpikeInterface channel ids.

    SpikeInterface 0.104's ``read_nwb_recording`` uses the raw NWB
    electrodes table's ``channel_name`` string column as the
    channel id if present; otherwise integer ``electrode_id`` is
    the channel id (the 1-1 fallback). Matches v1's lookup at
    ``v1/recording.py:683-712`` so production NWBs that carry a
    ``channel_name`` column resolve correctly. The Frank-lab
    MEArec fixture lacks the column and falls through to the
    integer path.
    """
    import pynwb

    from spyglass.common.common_nwbfile import Nwbfile

    nwb_file_abs_path = Nwbfile.get_abs_path(nwb_file_name)
    with pynwb.NWBHDF5IO(nwb_file_abs_path, mode="r") as io:
        nwbfile = io.read()
        electrodes_table = nwbfile.electrodes
        if "channel_name" not in electrodes_table.colnames:
            return [int(c) for c in spyglass_ids]
        channel_names = electrodes_table["channel_name"]
        return [channel_names[int(c)] for c in spyglass_ids]


def fetch_sort_group_probe_info(
    nwb_file_name: str, channel_ids
) -> tuple[tuple, tuple]:
    """Fetch per-channel ``probe_type`` + ``electrode_group_name``.

    Returns a pair of tuples (probe_types, electrode_group_names),
    one entry per channel id in ``channel_ids``. The tuple form
    is DeepHash-stable (NamedTuple field constraint for
    ``RecordingFetched``). Used by both the populate path
    (``make_fetch``) and the rebuild path
    (``_rebuild_nwb_artifact``) to feed
    ``maybe_apply_tetrode_geometry``.

    The fetch is ``order_by="electrode_id"`` so two successive
    ``make_fetch`` calls return byte-identical tuples; without an
    explicit ordering, DataJoint/MySQL row order is unspecified and
    the tri-part DeepHash integrity check inside the populate
    transaction can spuriously raise on reorder. This matches the
    ordered-fetch pattern used by the other tri-part ``make_fetch``
    paths in this package (e.g. ``ArtifactDetection.make_fetch`` in
    artifact.py, which orders its member fetch by ``recording_id``).
    """
    from spyglass.common.common_device import Probe as _Probe
    from spyglass.common.common_ephys import Electrode as _Electrode

    probe_rows = (
        _Electrode * _Probe
        & {"nwb_file_name": nwb_file_name}
        & [{"electrode_id": int(c)} for c in channel_ids]
    ).fetch(
        "probe_type",
        "electrode_group_name",
        as_dict=True,
        order_by="electrode_id",
    )
    probe_types = tuple(r["probe_type"] for r in probe_rows)
    electrode_group_names = tuple(r["electrode_group_name"] for r in probe_rows)
    return probe_types, electrode_group_names


def maybe_apply_tetrode_geometry(
    recording,
    probe_types: tuple,
    electrode_group_names: tuple,
    sort_group_channel_ids: list,
):
    """Attach the ``tetrode_12.5`` probe geometry when the sort group fits.

    Matches v1's patch at ``v1/recording.py:630-643``: sort groups
    of exactly 4 channels on a single ``tetrode_12.5`` probe and
    a single electrode group get an explicit
    ``(0,0)-(0,12.5)-(12.5,0)-(12.5,12.5)`` µm probe with 6.25 µm
    contact radius. Covers legacy Frank-lab NWBs where contact
    positions were never written into the electrode table.
    Geometry-aware sorters (Kilosort, MountainSort5) on those
    recordings depend on this patch; clusterless_thresholder and
    MS4 are unaffected.

    When any gate fails the recording is returned untouched and an
    ``INFO`` log names the failed gate's reason, so an operator
    debugging "Kilosort sees the wrong geometry" can grep the populate
    log for which condition skipped the patch.
    """
    from spyglass.utils import logger

    unique_probes = set(probe_types)
    unique_groups = set(electrode_group_names)
    # First failing gate wins; the reason text lives next to its
    # predicate so adding/removing a gate is a one-line edit with no
    # index alignment. ``next(iter(...), None)`` avoids StopIteration on
    # an empty probe set -- the ``len != 1`` gate above it fires first.
    gates = (
        (
            len(unique_probes) != 1,
            "sort group spans multiple probe types "
            "(expected a single tetrode_12.5)",
        ),
        (
            next(iter(unique_probes), None) != "tetrode_12.5",
            "single probe is not tetrode_12.5",
        ),
        (
            len(sort_group_channel_ids) != 4,
            "sort group does not have exactly 4 channels",
        ),
        (
            len(unique_groups) != 1,
            "sort group spans multiple electrode groups",
        ),
    )
    for failed, reason in gates:
        if failed:
            logger.info("_maybe_apply_tetrode_geometry skipped: %s", reason)
            return recording

    import numpy as _np
    import probeinterface as pi

    tetrode = pi.Probe(ndim=2)
    position = [[0, 0], [0, 12.5], [12.5, 0], [12.5, 12.5]]
    tetrode.set_contacts(
        position,
        shapes="circle",
        shape_params={"radius": 6.25},
    )
    tetrode.set_contact_ids([str(c) for c in sort_group_channel_ids])
    tetrode.set_device_channel_indices(_np.arange(4))
    return recording.set_probe(tetrode, in_place=True)


# Pitch-anchored adjacency for the ``interpolate`` re-inclusion. Constants are
# dimensionless multiples of the probe's own physical pitch, so one rule fits
# dense Neuropixels shanks and sparse polymer groups alike.
MIN_GOOD_NEIGHBORS = 2  # surrounded (>=2), not merely adjacent on one side
RADIUS_FACTOR = 1.5  # one pitch away counts; a multi-pitch gap does not


def _shank_pitch(shank_xyz):
    """Median nearest-neighbor distance over ALL electrodes on a shank.

    ``shank_xyz``: (M, 3) probe-relative positions (``Probe.Electrode``
    ``rel_x/rel_y/rel_z``) of every electrode on the shank (good AND bad), so
    the result is the probe's physical pitch, independent of which channels a
    sort group happens to keep. Returns ``None`` when the shank has < 2
    electrodes OR any coordinate is non-finite (``rel_x/rel_y/rel_z`` are
    nullable -> a NULL arrives as NaN); a ``None`` pitch makes the caller raise
    the clear "needs positions" error rather than silently producing NaN
    distances.
    """
    import numpy as np

    xyz = np.asarray(shank_xyz, dtype=float)
    if xyz.shape[0] < 2 or not np.isfinite(xyz).all():
        return None
    dd = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(dd, np.inf)
    return float(np.median(dd.min(axis=1)))


def _interior_bad_channel_ids(good_xyz, candidate_xyz, pitch):
    """Curated-bad ids physically embedded among a group's good channels.

    ``good_xyz``: (N, 3) probe-relative positions of the group's good channels.
    ``candidate_xyz``: list of ``(electrode_id, (rel_x, rel_y, rel_z))`` for the
    curated-bad electrodes on the group's shank(s). ``pitch``: the shank's
    physical electrode spacing from :func:`_shank_pitch` (NOT derived from the
    possibly-sparse good set). A candidate is kept only when at least
    ``MIN_GOOD_NEIGHBORS`` good channels lie within ``RADIUS_FACTOR * pitch`` of
    it -- so a bad channel between two far-apart good channels (its nearest good
    channel many pitches away) is excluded, while a bad channel in a dense run
    is kept. Returns a sorted list. Defensive on non-finite input
    (``make_fetch`` raises before calling): a non-finite ``pitch`` or good
    position -> ``[]``; a candidate with a non-finite position is skipped (never
    silently treated as adjacent).
    """
    import numpy as np

    good = np.asarray(good_xyz, dtype=float)
    if (
        good.shape[0] < 2
        or not pitch
        or not np.isfinite(pitch)
        or not np.isfinite(good).all()
    ):
        return []  # need >=2 finite good channels and a finite pitch
    radius = RADIUS_FACTOR * float(pitch)
    return sorted(
        int(cid)
        for cid, pos in candidate_xyz
        if np.isfinite(pos).all()
        and int(
            (
                np.linalg.norm(good - np.asarray(pos, float), axis=1) <= radius
            ).sum()
        )
        >= MIN_GOOD_NEIGHBORS
    )


def fetch_interior_bad_channel_ids(
    nwb_file_name: str, sort_group_channel_ids
) -> tuple:
    """The sort group's *interior* curated-bad electrode ids (interpolate path).

    Returns a sorted tuple of the curated-bad (``Electrode.bad_channel='True'``)
    electrodes on the sort group's shank(s) that are physically embedded among
    the group's good channels (>= ``MIN_GOOD_NEIGHBORS`` good channels within
    ``RADIUS_FACTOR * pitch``, ``pitch`` the full-shank spacing). These are the
    channels ``interpolate`` re-includes and fills. The candidate set is scoped
    to the group's ``(electrode_group_name, probe_shank)`` -- NOT a shank-wide
    grab -- because ``set_group_by_electrode_table_column`` builds
    arbitrary-membership groups and stores no original column values.

    Geometry comes from the probe-relative ``Probe.Electrode``
    ``rel_x/rel_y/rel_z`` (joined onto ``Electrode``), NOT ``Electrode.x/y/z``:
    the latter are absolute brain coordinates that are commonly unset (all
    zero / NULL), whereas ``rel_*`` is the physical contact geometry -- the
    same coordinate system SpikeInterface's ``get_channel_locations`` (and
    therefore ``interpolate_bad_channels``) reads from the NWB. Using the same
    source keeps the adjacency decision consistent with the actual fill.

    Raises ``ValueError`` (pointing the user at ``remove``) when any required
    position is null/NaN, a needed electrode has no probe geometry, or a shank's
    pitch is undefined -- the full-shank position fetch is a superset of the
    good + candidate positions, so a ``_shank_pitch`` of ``None`` is the
    finiteness gate for the whole shank (rather than silently returning an empty
    set that masquerades as "no bad channels to fill"). DB-touching at call time
    via lazy ``Electrode`` / ``Probe`` imports; the geometry math is delegated to
    the pure helpers above.
    """
    import numpy as np

    from spyglass.common.common_device import Probe
    from spyglass.common.common_ephys import Electrode

    def _fail(what: str):
        raise ValueError(
            "Recording.make: bad_channel_handling='interpolate' needs probe "
            f"geometry for {nwb_file_name!r}, but {what}. Use "
            "bad_channel_handling='remove', or populate Probe.Electrode "
            "rel_x/rel_y/rel_z and probe_shank for the sort group's shank(s)."
        )

    good_ids = {int(c) for c in sort_group_channel_ids}
    # Inner-join Electrode with Probe.Electrode to bring in rel_x/rel_y/rel_z
    # (and probe_shank). Electrodes lacking a probe link drop out of the join,
    # so an incomplete result means a member without geometry -> fail loud.
    good_rows = (
        (Electrode * Probe.Electrode)
        & {"nwb_file_name": nwb_file_name}
        & [{"electrode_id": c} for c in sorted(good_ids)]
    ).fetch(
        "electrode_id",
        "electrode_group_name",
        "probe_shank",
        "rel_x",
        "rel_y",
        "rel_z",
        as_dict=True,
    )
    found_ids = {int(r["electrode_id"]) for r in good_rows}
    if found_ids != good_ids:
        missing = sorted(good_ids - found_ids)
        _fail(
            f"sort-group electrode(s) {missing[:5]} have no probe geometry "
            "(no Probe.Electrode link)"
        )

    # Group the good channels by physical shank.
    good_by_shank: dict[tuple[str, int], list] = {}
    for r in good_rows:
        if r["probe_shank"] is None:
            _fail(f"electrode {int(r['electrode_id'])} has no probe_shank")
        pos = np.array([r["rel_x"], r["rel_y"], r["rel_z"]], dtype=float)
        if not np.isfinite(pos).all():
            _fail(
                f"good electrode {int(r['electrode_id'])} has a null position"
            )
        good_by_shank.setdefault(
            (str(r["electrode_group_name"]), int(r["probe_shank"])), []
        ).append(pos)

    interior: list[int] = []
    for (egroup, shank), good_xyz in good_by_shank.items():
        restr = {
            "nwb_file_name": nwb_file_name,
            "electrode_group_name": egroup,
            "probe_shank": shank,
        }
        # Full-shank positions -> physical pitch (and the finiteness gate).
        shank_xyz = np.array(
            [
                [r["rel_x"], r["rel_y"], r["rel_z"]]
                for r in ((Electrode * Probe.Electrode) & restr).fetch(
                    "rel_x", "rel_y", "rel_z", as_dict=True
                )
            ],
            dtype=float,
        )
        pitch = _shank_pitch(shank_xyz)
        if pitch is None:
            _fail(
                f"shank {shank} of group {egroup!r} has < 2 positioned "
                "electrodes or a null coordinate (pitch undefined)"
            )
        # Curated-bad candidates on this shank, excluding any that are already
        # sort-group members (a remove_bad_channels=False group keeps its own
        # bad members present; only the *excluded* interior bad are re-included).
        candidate_xyz = [
            (
                int(r["electrode_id"]),
                np.array([r["rel_x"], r["rel_y"], r["rel_z"]], dtype=float),
            )
            for r in (
                (Electrode * Probe.Electrode) & restr & {"bad_channel": "True"}
            ).fetch("electrode_id", "rel_x", "rel_y", "rel_z", as_dict=True)
            if int(r["electrode_id"]) not in good_ids
        ]
        interior.extend(
            _interior_bad_channel_ids(good_xyz, candidate_xyz, pitch)
        )
    return tuple(sorted(set(interior)))


def apply_pre_motion_preprocessing(
    recording,
    reference_mode: str,
    reference_electrode_id: int | None,
    sort_group_channel_ids: list,
    validated,
    bad_channel_handling: str = "remove",
    bad_channel_ids=(),
):
    """Apply the pre-motion preprocessing stack (phase-shift, filter, ref).

    Optional ADC phase-shift runs FIRST, then the bandpass filter
    (temporal), then referencing (spatial). Bandpass-before-reference is
    the signal-processing-preferred order and intentionally diverges from
    v1, which referenced first. The two are not commutative on the
    global-median branch (the per-sample median is non-linear), so v2 and v1
    differ numerically there; on the ``specific`` / ``none`` paths the steps
    are linear and commute, so output is identical to the old order.
    Whitening stays deferred to the sorter stage so motion correction never
    sees whitened data (SpikeInterface docs flag whitening as destructive for
    motion estimators).

    Takes a pre-validated ``PreprocessingParamsSchema`` instance
    so the DB read happens once in ``make_fetch`` -- this method
    is called from inside ``make_compute`` where the tri-part
    contract forbids further DB I/O.

    Returns ``(recording, applied_steps)`` where ``applied_steps`` reports the
    runtime facts a caller cannot derive from the params alone -- currently
    ``{"phase_shift": bool}``, whether the gated phase-shift actually ran (it
    is skipped when the recording lacks ``inter_sample_shift``).
    ``filtering_description`` consumes it so provenance can distinguish a
    phase-shift that ran from one that was requested but skipped. Bandpass and
    reference have no skip path (they run iff their params are set), so they
    are not reported -- the params are ground truth for them.
    """
    import numpy as _np
    import spikeinterface.preprocessing as sip

    from spyglass.utils import logger

    applied_steps: dict = {}

    # 0. ADC phase-shift (multiplexed-ADC / Neuropixels) -- runs FIRST,
    #    before the bandpass. Valid only when the recording carries an
    #    ``inter_sample_shift`` property (``sip.phase_shift`` raises without
    #    it); skip (loudly) otherwise so enabling it on non-multiplexed
    #    acquisition is a safe no-op rather than a hard failure. The AIND
    #    "phase-shift first" order is a deliberate choice, not a correctness
    #    requirement (a fractional-sample delay and a zero-phase bandpass are
    #    both LTI per channel and commute, modulo edge margins).
    applied_steps["phase_shift"] = False
    if validated.phase_shift is not None:
        if recording.get_property("inter_sample_shift") is not None:
            recording = sip.phase_shift(
                recording, margin_ms=validated.phase_shift.margin_ms
            )
            applied_steps["phase_shift"] = True
        else:
            logger.warning(
                "apply_pre_motion_preprocessing: phase_shift requested but the "
                "recording has no 'inter_sample_shift' property (not a "
                "multiplexed-ADC acquisition); skipping phase-shift."
            )

    # 1. Bandpass filter. ``bandpass_filter=None`` is the "no_filter"
    #    preset -- skip the step entirely rather than passing a wide band
    #    that still filters.
    if validated.bandpass_filter is not None:
        recording = sip.bandpass_filter(
            recording,
            freq_min=validated.bandpass_filter.freq_min,
            freq_max=validated.bandpass_filter.freq_max,
            dtype=_np.float64,
        )

    # 1b. Bad-channel handling: between filter and reference (matches the
    #     IBL/AIND destripe order). Only ``interpolate`` does anything -- it
    #     fills the interior curated-bad channels ``restrict_recording``
    #     re-included. On ``remove`` those channels were never re-added, so
    #     ``bad_channel_ids`` is empty here and this is a no-op (today's
    #     behavior). The ``specific`` reference is present only for subtraction
    #     (dropped after ``common_reference``) and is never a handling target.
    present = {int(c) for c in recording.get_channel_ids()}
    ref_id = (
        int(reference_electrode_id) if reference_mode == "specific" else None
    )
    to_interpolate = sorted(
        (present & {int(c) for c in bad_channel_ids}) - {ref_id}
    )
    if bad_channel_handling == "interpolate" and to_interpolate:
        # interpolation needs channel locations; fail clearly if absent rather
        # than letting SI raise opaquely deep in the kriging call.
        if recording.get_channel_locations() is None:
            raise ValueError(
                "apply_pre_motion_preprocessing: interpolate requires channel "
                "locations on the recording, but none are set (no probe "
                "geometry). Use bad_channel_handling='remove', or ensure the "
                "NWB / probe carries electrode positions."
            )
        recording = sip.interpolate_bad_channels(recording, to_interpolate)
    else:
        to_interpolate = []
    applied_steps["bad_channels"] = {"interpolated": to_interpolate}

    # 2. Reference the filtered signal.
    if reference_mode == "specific":
        recording = sip.common_reference(
            recording,
            reference="single",
            ref_channel_ids=[int(reference_electrode_id)],
            dtype=_np.float64,
        )
        # Drop the reference channel from the recording surface so
        # the sorter only sees the actual sort-group channels
        # (restrict_recording included it solely for this step).
        # Its presence here is an invariant: restrict_recording always
        # slices the specific reference in, and bandpass / common_reference
        # preserve channel ids, so a missing ref means an upstream contract
        # broke. Fail loud rather than silently shipping a reference channel
        # into the sort.
        channel_ids = [int(c) for c in recording.get_channel_ids()]
        if int(reference_electrode_id) not in channel_ids:
            raise RuntimeError(
                "apply_pre_motion_preprocessing: 'specific' reference "
                f"electrode {int(reference_electrode_id)} is absent after "
                f"referencing (channels={channel_ids}); cannot drop it from "
                "the sort surface. restrict_recording must slice the "
                "reference channel in for subtraction."
            )
        recording = recording.remove_channels([int(reference_electrode_id)])
    elif reference_mode == "global_median":
        # A global reference on a single-channel group zeroes the signal: the
        # per-sample median (or mean) across one channel IS that channel, so
        # subtracting it yields all zeros -- a recording that sorts to nothing
        # with no error. Fail loud instead.
        n_channels = len(recording.get_channel_ids())
        if n_channels < 2:
            raise ValueError(
                "apply_pre_motion_preprocessing: 'global_median' reference on "
                f"a {n_channels}-channel sort group zeroes the signal (the "
                "median/mean across one channel is that channel itself). Use "
                "reference_mode='none' for unitrodes, or omit_unitrode=True "
                "when creating the sort group."
            )
        recording = sip.common_reference(
            recording,
            reference="global",
            operator=validated.common_reference.operator,
            dtype=_np.float64,
        )
    elif reference_mode != "none":
        raise ValueError(
            "Recording.make: invalid reference_mode "
            f"{reference_mode!r}. Use 'none', 'global_median', or "
            "'specific'."
        )
    return recording, applied_steps


def filtering_description(
    bandpass_filter, reference_mode: str, applied_steps: dict
) -> str:
    """``ElectricalSeries.filtering`` provenance from steps ACTUALLY run.

    Built from the preprocessing that actually ran so the persisted NWB
    metadata does not claim a step that did not happen -- e.g. the
    ``no_filter`` preset (``bandpass_filter`` None), ``reference_mode='none'``,
    or a phase-shift that was requested but skipped because the recording
    lacks an ``inter_sample_shift`` property. The phase-shift claim is driven
    by ``applied_steps`` (the report from ``apply_pre_motion_preprocessing``),
    not the params, precisely so a requested-but-skipped phase-shift is not
    falsely listed. Important for archival / DANDI export; the string is
    descriptive only and is not read back internally. Steps are listed in the
    order the runtime APPLIES them -- phase-shift first, then bandpass filter,
    then bad-channel interpolation, then common reference (see
    ``apply_pre_motion_preprocessing``) -- since that order is non-commutative
    on the global-median branch.

    The bad-channel interpolation claim is driven by ``applied_steps`` (the
    actual count that ran), not the params, and is appended ONLY when that count
    is > 0 -- so the default ``remove`` path (count 0) leaves the string
    byte-for-byte unchanged (this string is persisted into
    ``ElectricalSeries.filtering`` and hashed into ``cache_hash``).
    """
    steps = []
    if applied_steps.get("phase_shift"):
        steps.append("phase-shift (ADC)")
    if bandpass_filter is not None:
        steps.append(
            f"bandpass filter {bandpass_filter.freq_min:g}-"
            f"{bandpass_filter.freq_max:g} Hz"
        )
    n_interp = len(
        applied_steps.get("bad_channels", {}).get("interpolated", [])
    )
    if n_interp:
        steps.append(
            f"interpolate {n_interp} bad channel{'s' if n_interp != 1 else ''}"
        )
    if reference_mode != "none":
        steps.append(f"common reference ({reference_mode})")
    return "; ".join(steps) if steps else "none (raw, no preprocessing)"


def write_nwb_artifact(
    recording,
    nwb_file_name: str,
    existing_analysis_file_name: str | None = None,
    timestamps_override=None,
    *,
    filtering_description: str,
) -> tuple[str, str, str]:
    """Write the preprocessed recording into an ``AnalysisNwbfile``.

    Streams the ``(n_samples, n_channels)`` trace array and the
    ``(n_samples,)`` timestamps vector into the ElectricalSeries
    via HDMF's ``GenericDataChunkIterator`` (``buffer_gb=5``,
    matching v1's production choice). Without streaming, a
    30 kHz x 128 ch x 1 h recording (~110 GB float64) would have
    to materialize in RAM before the NWB write, which OOMs on
    any lab workstation.

    Returns ``(analysis_file_name, electrical_series_object_id,
    cache_hash)``. The ``cache_hash`` is computed **after** the
    write via ``_hash_nwb_recording`` -- the ``NwbfileHasher``
    digest of the file we just persisted. The v1 recompute machinery
    uses the same hashing path, so v2 verification does not maintain a
    parallel implementation.

    Writes the file to disk only; the caller registers the
    ``AnalysisNwbfile`` row inside its DataJoint transaction so
    the file registration and the v2 row commit atomically.

    Parameters
    ----------
    recording : si.BaseRecording
        The preprocessed recording to materialize.
    nwb_file_name : str
        Parent NWB filename (passed to ``AnalysisNwbfile().create``).
    existing_analysis_file_name : str, optional
        When set, write into the existing slot (the recompute /
        rebuild path) rather than minting a new analysis file.
    timestamps_override : numpy.ndarray, optional
        Pre-computed persisted timestamps. ``None`` lets the helper
        concatenate per-segment
        ``recording.get_times()`` via
        :func:`_get_recording_timestamps`.
    """
    import numpy as _np
    import pynwb

    from spyglass.spikesorting.v2._nwb_iterators import (
        SpikeInterfaceRecordingDataChunkIterator,
        TimestampsDataChunkIterator,
    )
    from spyglass.spikesorting.v2.utils import (
        _get_recording_timestamps,
        _hash_nwb_recording,
        electrode_table_region,
        resolve_conversion_and_offset,
        write_buffer_gb,
    )

    import pathlib as _pathlib

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import _ELECTRICAL_SERIES_NAME
    from spyglass.utils import logger

    # ``AnalysisNwbfile().create`` writes a stub file to disk
    # before we open it for the streaming write. Track the
    # filename from the first byte on disk and unlink on any
    # failure between here and the post-write hash, so a
    # partial / aborted write never outlives this call.
    analysis_file_name = AnalysisNwbfile().create(
        nwb_file_name=nwb_file_name,
        recompute_file_name=existing_analysis_file_name,
    )
    try:
        analysis_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name,
            from_schema=bool(existing_analysis_file_name),
        )

        # Traces are written unscaled (return_in_uV=False), so the
        # ElectricalSeries must carry gain (as ``conversion``) AND offset
        # (as ``offset``) to recover real volts on readback:
        # ``volts = raw * conversion + offset``. The resolver rejects
        # heterogeneous gain/offset and non-positive gain (v1 silently
        # picked gains[0] and dropped offset entirely).
        conversion, es_offset = resolve_conversion_and_offset(recording)

        # The data iterator drives ``recording.get_traces(...)``
        # per chunk and never materializes the whole array. The
        # timestamps iterator wraps a 1D vector; resolve through
        # ``_get_recording_timestamps`` so multi-segment NWBs and
        # persisted-timestamps overrides both flow through correctly.
        sampling_frequency = float(recording.get_sampling_frequency())
        timestamps = _get_recording_timestamps(
            recording, override=timestamps_override
        )
        # Bound the buffers to ~a fixed duration of data so a narrow sort
        # group (e.g. a 4-ch tetrode) does not buffer the whole recording
        # in one 5 GB chunk; wide groups stay capped at 5 GB.
        data_iterator = SpikeInterfaceRecordingDataChunkIterator(
            recording=recording,
            return_in_uV=False,
            buffer_gb=write_buffer_gb(
                recording.get_num_channels(), sampling_frequency
            ),
        )
        timestamps_iterator = TimestampsDataChunkIterator(
            timestamps=timestamps,
            sampling_frequency=sampling_frequency,
            buffer_gb=write_buffer_gb(1, sampling_frequency),
        )

        with pynwb.NWBHDF5IO(
            path=analysis_abs_path, mode="a", load_namespaces=True
        ) as io:
            nwbfile = io.read()
            # ``recording.get_channel_ids()`` are spyglass electrode ids;
            # map them to electrodes-table ROW INDICES (not raw ids) so a
            # non-contiguous / reordered electrodes table does not silently
            # mis-point the ElectricalSeries at the wrong electrodes.
            table_region = electrode_table_region(
                nwbfile,
                recording.get_channel_ids(),
                "Sort group electrodes",
            )
            series = pynwb.ecephys.ElectricalSeries(
                name=_ELECTRICAL_SERIES_NAME,
                data=data_iterator,
                electrodes=table_region,
                timestamps=timestamps_iterator,
                filtering=filtering_description,
                description=(
                    f"Pre-motion preprocessed recording from "
                    f"{nwb_file_name} for spike sorting"
                ),
                conversion=conversion,
                offset=es_offset,
            )
            nwbfile.add_acquisition(series)
            object_id = nwbfile.acquisition[_ELECTRICAL_SERIES_NAME].object_id
            io.write(nwbfile)

        # Hash the persisted file (not in-memory bytes) so the
        # digest reflects what was actually written --
        # timestamps, electrodes, conversion, ElectricalSeries
        # metadata -- not just trace data. Matches the v1
        # recompute hashing path.
        cache_hash = _hash_nwb_recording(analysis_file_name)
    except Exception:
        try:
            _abs = AnalysisNwbfile.get_abs_path(
                analysis_file_name,
                from_schema=bool(existing_analysis_file_name),
            )
            _pathlib.Path(_abs).unlink(missing_ok=True)
        except Exception as cleanup_exc:  # pragma: no cover -- defensive
            logger.error(
                "Recording._write_nwb_artifact: failed to clean up "
                f"partial analysis file {analysis_file_name!r}: "
                f"{cleanup_exc!r}"
            )
        raise

    return analysis_file_name, object_id, cache_hash
