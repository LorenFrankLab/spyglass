"""Pure helper tests and DB-free NWB integration tests for concatenation.

Drive ``_concat_recording`` directly -- the sample-boundary / spike-train
back-mapping math is pure. Synthetic recordings exercise in-memory stitching;
NWB integration cases exercise persisted clocks, scaling, masking, and stitching.
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.spikesorting.v2._concat_recording import (
    build_concatenated_recording,
    cumulative_member_boundaries,
    member_set_hash,
    member_spike_times,
    member_split_key,
    split_spike_frames_by_spans,
    split_unit_spike_trains,
)
from spyglass.spikesorting.v2.exceptions import ConcatSplitError

# ---------- cumulative_member_boundaries -----------------------------------


@pytest.mark.unit
def test_cumulative_boundaries_basic():
    """Boundaries are running totals; the last equals the grand total."""
    assert cumulative_member_boundaries([100, 50, 30]) == [100, 150, 180]


@pytest.mark.unit
def test_cumulative_boundaries_empty():
    """No members -> no boundaries."""
    assert cumulative_member_boundaries([]) == []


# ---------- split_unit_spike_trains ----------------------------------------


@pytest.mark.unit
def test_split_maps_to_local_member_frames():
    """Each member's spikes are shifted into that member's local frame and the
    boundary frame (== end) belongs to the NEXT member."""
    # Two members of 100 samples each; boundaries [100, 200].
    trains = {
        7: np.array([0, 50, 99, 100, 150, 199]),
        9: np.array([10, 100]),  # 100 is the first frame of member 1
    }
    per_member = split_unit_spike_trains(trains, [100, 200])
    assert len(per_member) == 2
    # Member 0 keeps frames in [0, 100): unchanged.
    np.testing.assert_array_equal(per_member[0][7], [0, 50, 99])
    np.testing.assert_array_equal(per_member[0][9], [10])
    # Member 1 keeps frames in [100, 200) shifted by -100.
    np.testing.assert_array_equal(per_member[1][7], [0, 50, 99])
    np.testing.assert_array_equal(per_member[1][9], [0])


@pytest.mark.unit
def test_member_spike_times_keep_each_members_own_clock():
    """Two members with gapped clocks of unequal length: frames split by the
    member spans map onto each member's own timestamps (hand-computed), and
    a unit absent from a member keeps an empty train."""
    member0 = 100.0 + np.arange(4) * 0.5  # 100.0 .. 101.5
    member1 = 250.0 + np.arange(3) * 0.5  # 250.0 .. 251.0 (after a gap)
    per_member = split_spike_frames_by_spans(
        {0: np.array([1, 3, 4, 6]), 5: np.array([5])}, [(0, 4), (4, 7)]
    )
    assert [
        {
            u: t.tolist()
            for u, t in member_spike_times(f, ts, context="x").items()
        }
        for f, ts in zip(per_member, (member0, member1))
    ] == [{0: [100.5, 101.5], 5: []}, {0: [250.0, 251.0], 5: [250.5]}]


@pytest.mark.unit
def test_member_spike_times_rejects_frames_outside_the_member():
    """A local frame past the member's last sample raises, naming the
    caller's context."""
    with pytest.raises(ValueError, match="member 2 of x: local spike frames"):
        member_spike_times(
            {0: np.array([0, 3])}, np.arange(3.0), context="member 2 of x"
        )


@pytest.mark.unit
def test_member_split_key_disambiguates_same_spatial_member():
    """The split key is the full member identity (nwb, sort_group, interval,
    team), so two members sharing nwb+interval but differing in sort group OR
    team get distinct keys -- a lossy (nwb, interval) key would collide them."""
    member = {
        "nwb_file_name": "day1_.nwb",
        "sort_group_id": 0,
        "interval_list_name": "raw data valid times",
        "team_name": "team_a",
    }
    assert member_split_key(member) == (
        "day1_.nwb",
        0,
        "raw data valid times",
        "team_a",
    )
    # Same NWB + interval, different sort group -> distinct.
    other_group = {**member, "sort_group_id": 1}
    assert member_split_key(member) != member_split_key(other_group)
    # Same NWB + interval + sort group, different team (allowed mixed-team
    # member) -> still distinct, not silently overwritten.
    other_team = {**member, "team_name": "team_b"}
    assert member_split_key(member) != member_split_key(other_team)
    assert (
        len(
            {
                member_split_key(member),
                member_split_key(other_group),
                member_split_key(other_team),
            }
        )
        == 3
    )


@pytest.mark.unit
def test_split_preserves_unit_ids_with_empty_arrays():
    """A unit absent from a member's span still appears, with an empty array."""
    trains = {3: np.array([5]), 4: np.array([150])}
    per_member = split_unit_spike_trains(trains, [100, 200])
    assert set(per_member[0]) == {3, 4}
    assert set(per_member[1]) == {3, 4}
    assert per_member[0][4].size == 0
    assert per_member[1][3].size == 0


@pytest.mark.unit
def test_split_conserves_every_spike_per_unit():
    """Per-spike conservation: every input (unit, frame) lands in exactly one
    member and back-maps to its original frame -- not just matching summed counts
    (which a simultaneous drop + duplicate could satisfy)."""
    boundaries = [100, 250, 400]
    trains = {
        7: np.array([0, 99, 100, 249, 250, 399]),
        9: np.array([5, 150, 399]),
        11: np.array([], dtype=np.int64),
    }
    per_member = split_unit_spike_trains(trains, boundaries)

    starts = [0, 100, 250]
    for unit_id, original in trains.items():
        # Reconstruct the global frames from each member's local frames.
        reconstructed = np.sort(
            np.concatenate(
                [
                    per_member[m][unit_id] + starts[m]
                    for m in range(len(boundaries))
                ]
            )
        ).astype(np.int64)
        np.testing.assert_array_equal(reconstructed, np.sort(original))


@pytest.mark.unit
def test_split_raises_when_a_spike_is_past_the_final_boundary():
    """A frame at/after the final boundary belongs to no member; rather than
    silently dropping it, the split raises ``ConcatSplitError``."""
    trains = {7: np.array([0, 50, 200])}  # 200 == final boundary, out of range
    with pytest.raises(ConcatSplitError, match="outside"):
        split_unit_spike_trains(trains, [100, 200])


@pytest.mark.unit
def test_split_raises_on_negative_spike_frame():
    """A negative frame is assigned to no member; conservation raises."""
    trains = {7: np.array([-1, 10, 150])}
    with pytest.raises(ConcatSplitError, match="outside"):
        split_unit_spike_trains(trains, [100, 200])


@pytest.mark.unit
def test_split_raises_on_non_strictly_increasing_boundaries():
    """Equal/decreasing boundaries make the member intervals overlap or empty,
    so a spike could land in two members (or none); reject them."""
    trains = {7: np.array([10, 120])}
    with pytest.raises(ConcatSplitError, match="strictly increasing"):
        split_unit_spike_trains(trains, [100, 100, 200])


@pytest.mark.unit
def test_split_raises_when_final_boundary_below_total_sample_count():
    """When the caller passes the concat sample count, a boundary set that does
    not cover the full recording is rejected before any spike is dropped."""
    trains = {7: np.array([10, 150])}
    with pytest.raises(ConcatSplitError, match="sample count"):
        split_unit_spike_trains(trains, [100, 200], total_n_samples=300)


# ---------- member_set_hash ------------------------------------------------


def _snap(member_index, recording_id, content_hash="c" * 64, **over):
    """Build a member-snapshot row for the hash tests."""
    row = {
        "member_index": member_index,
        "nwb_file_name": "day1_.nwb",
        "sort_group_id": 0,
        "interval_list_name": "raw data valid times",
        "team_name": "team_a",
        "recording_id": recording_id,
        "recording_content_hash": content_hash,
        "artifact_detection_id": None,
    }
    row.update(over)
    return row


@pytest.mark.unit
def test_member_set_hash_is_a_stable_sha256_hex():
    """The folded member-set hash is a deterministic 64-char hex digest, stable
    across calls and insertion order (canonicalized by member_index)."""
    rows = [
        _snap(0, "11111111-1111-1111-1111-111111111111"),
        _snap(1, "22222222-2222-2222-2222-222222222222"),
    ]
    h1 = member_set_hash(rows)
    h2 = member_set_hash(list(reversed(rows)))  # input order must not matter
    assert h1 == h2
    assert len(h1) == 64
    assert all(ch in "0123456789abcdef" for ch in h1)


@pytest.mark.unit
def test_member_set_hash_changes_with_member_identity():
    """A different member (different recording_id) yields a different hash --
    this is what makes a different member SET a different concat id."""
    base = [
        _snap(0, "11111111-1111-1111-1111-111111111111"),
        _snap(1, "22222222-2222-2222-2222-222222222222"),
    ]
    swapped = [
        _snap(0, "11111111-1111-1111-1111-111111111111"),
        _snap(1, "33333333-3333-3333-3333-333333333333"),
    ]
    assert member_set_hash(base) != member_set_hash(swapped)


@pytest.mark.unit
def test_member_set_hash_changes_with_member_order():
    """Re-assigning member_index (changing the concatenation order) changes the
    hash -- order is load-bearing for the stitched timeline."""
    rid_a = "11111111-1111-1111-1111-111111111111"
    rid_b = "22222222-2222-2222-2222-222222222222"
    forward = [_snap(0, rid_a), _snap(1, rid_b)]
    reordered = [_snap(0, rid_b), _snap(1, rid_a)]
    assert member_set_hash(forward) != member_set_hash(reordered)


@pytest.mark.unit
@pytest.mark.parametrize(
    "field,value",
    [
        ("nwb_file_name", "another_day_.nwb"),
        ("sort_group_id", 7),
        ("interval_list_name", "another interval"),
        ("team_name", "another team"),
        ("artifact_detection_id", "33333333-3333-3333-3333-333333333333"),
    ],
)
def test_member_set_hash_changes_with_other_frozen_logical_fields(field, value):
    """Source ownership, selected interval and artifact mask are identity-bearing."""
    first = _snap(0, "11111111-1111-1111-1111-111111111111")
    second = _snap(1, "22222222-2222-2222-2222-222222222222")
    original = member_set_hash([first, second])
    assert member_set_hash([first, {**second, field: value}]) != original


@pytest.mark.unit
def test_member_set_hash_ignores_content_hash():
    """Per-member ``recording_content_hash`` is verification, not identity, so it
    must NOT change the folded set hash (content drift is caught separately)."""
    rid_a = "11111111-1111-1111-1111-111111111111"
    rid_b = "22222222-2222-2222-2222-222222222222"
    rows = [
        _snap(0, rid_a, content_hash="a" * 64),
        _snap(1, rid_b, content_hash="b" * 64),
    ]
    drifted = [
        _snap(0, rid_a, content_hash="f" * 64),
        _snap(1, rid_b, content_hash="e" * 64),
    ]
    assert member_set_hash(rows) == member_set_hash(drifted)


@pytest.mark.unit
def test_member_set_hash_normalizes_uuid_and_int_forms():
    """A UUID object vs its str form, and a numpy-like sort_group_id vs int,
    collapse to one hash so a fetched row and a freshly built row agree."""
    import uuid

    rid = uuid.UUID("11111111-1111-1111-1111-111111111111")
    as_obj = [_snap(0, rid, sort_group_id=np.int64(0))]
    as_str = [_snap(0, str(rid), sort_group_id=0)]
    assert member_set_hash(as_obj) == member_set_hash(as_str)


# ---------- build_concatenated_recording -----------------------------------


@pytest.mark.unit
def test_build_concatenated_recording_returns_members_traces_unchanged():
    """The members are stitched into one segment whose traces are exactly the
    members' traces in order: the sample count is the sum, the channels are
    preserved, and no motion correction (or any other transform) runs."""
    import spikeinterface as si

    fs = 30_000.0
    rng = np.random.default_rng(0)
    traces_a = rng.normal(0, 1, size=(300, 4)).astype(np.float32)
    traces_b = rng.normal(0, 1, size=(200, 4)).astype(np.float32)
    rec_a = si.NumpyRecording([traces_a], sampling_frequency=fs)
    rec_b = si.NumpyRecording([traces_b], sampling_frequency=fs)
    concat = build_concatenated_recording([rec_a, rec_b])
    assert concat.get_num_segments() == 1
    assert concat.get_num_samples() == 500
    assert list(concat.get_channel_ids()) == list(rec_a.get_channel_ids())
    np.testing.assert_array_equal(
        concat.get_traces(), np.concatenate([traces_a, traces_b], axis=0)
    )


# ---------- assert_concat_compatible ---------------------------------------


def _rec_with_locations(n_samples, channel_ids, locations, fs=30_000.0):
    """A synthetic NumpyRecording with explicit channel ids and geometry."""
    import spikeinterface as si

    rec = si.NumpyRecording(
        [np.zeros((n_samples, len(channel_ids)), dtype=np.float32)],
        sampling_frequency=fs,
        channel_ids=channel_ids,
    )
    rec.set_dummy_probe_from_locations(np.asarray(locations, dtype=float))
    return rec


@pytest.mark.unit
def test_assert_concat_compatible_accepts_matching_members():
    """Members sharing channel ids and geometry pass the pre-concat check."""
    from spyglass.spikesorting.v2._concat_recording import (
        assert_concat_compatible,
    )

    locs = [[0.0, 0.0], [0.0, 20.0]]
    a = _rec_with_locations(100, [1, 2], locs)
    b = _rec_with_locations(50, [1, 2], locs)
    assert_concat_compatible([a, b])  # no raise


def _persisted_concat_member(
    tmp_path, index, start_s, fs=30_000.0, *, clock="timestamps"
):
    """Write a calibrated member; return its reader and independent source arrays."""
    from spyglass.spikesorting.v2._recording_nwb import read_recording_nwb
    from tests.spikesorting.v2._ingest_helpers import (
        write_processed_recording_nwb,
    )

    traces = np.arange(4000, dtype=np.float32).reshape(2000, 2) + index
    timestamps = start_s + np.arange(len(traces)) / fs
    path, series_path = write_processed_recording_nwb(
        tmp_path / f"member_{index}.nwb",
        traces=traces,
        timestamps=timestamps,
        rel_positions=[[1, 2], [3, 22]],
        channel_ids=[10, 30],
        conversion=2e-6,
        offset=-7e-6,
    )
    if clock == "rate":
        import h5py

        with h5py.File(path, "a") as handle:
            series = handle[series_path]
            del series["timestamps"]
            starting_time = series.create_dataset(
                "starting_time", data=float(start_s)
            )
            starting_time.attrs["rate"] = fs
            starting_time.attrs["unit"] = "seconds"
    return (
        read_recording_nwb(path, electrical_series_path=series_path),
        traces,
        timestamps,
    )


@pytest.mark.parametrize(
    "offsets",
    [(0, 17), (17, 0), (0, 10_000), (10_000, 0), (0, -10_000), (-10_000, 0)],
    ids=[
        "small_compared",
        "small_anchor",
        "large_compared",
        "large_anchor",
        "negative_compared",
        "negative_anchor",
    ],
)
@pytest.mark.integration
@pytest.mark.nwb
@pytest.mark.io_heavy
def test_concat_persisted_timestamp_rates_preserve_members(
    tmp_path, monkeypatch, offsets
):
    """Offset roundoff permits stitching without changing source data or clocks."""
    from spyglass.spikesorting.v2._concat_recording import (
        mask_member_recordings,
    )

    members = [
        _persisted_concat_member(tmp_path, index, start)
        for index, start in enumerate(offsets)
    ]
    recordings = [recording for recording, _, _ in members]
    rates = [recording.get_sampling_frequency() for recording in recordings]
    assert (
        abs(rates[0] - rates[1]) > 1e-9
    )  # Exercise inferred-rate disagreement.

    def reject_whole_clock_read(*args, **kwargs):
        raise AssertionError(
            "Concat must not materialize a member's full clock"
        )

    for recording in recordings:
        monkeypatch.setattr(recording, "get_times", reject_whole_clock_read)
    first_times = members[0][2]
    valid_times = [
        [
            [first_times[0], first_times[100]],
            [first_times[200], first_times[-1]],
        ],
        None,
    ]
    masked, ranges = mask_member_recordings(recordings, valid_times)
    assert ranges == [(100, 200)]
    for recording in masked:
        monkeypatch.setattr(recording, "get_times", reject_whole_clock_read)
    concatenated = build_concatenated_recording(masked)

    expected_uv = np.concatenate([traces * 2 - 7 for _, traces, _ in members])
    expected_uv[100:200] = 0
    assert concatenated.get_num_segments() == 1
    assert concatenated.get_num_samples() == 4000
    assert concatenated.get_dtype() == np.dtype("float32")
    np.testing.assert_array_equal(concatenated.channel_ids, [10, 30])
    np.testing.assert_array_equal(
        concatenated.get_channel_locations(), [[1, 2], [3, 22]]
    )
    np.testing.assert_array_equal(concatenated.get_channel_gains(), [1, 1])
    np.testing.assert_array_equal(concatenated.get_channel_offsets(), [0, 0])
    np.testing.assert_array_equal(concatenated.get_traces(), expected_uv)
    np.testing.assert_array_equal(
        concatenated.get_traces(return_in_uV=True), expected_uv
    )
    # Independently compare with the known acquisition grid: the output starts
    # at zero and the tiny inferred-rate error stays below 0.01 sample here.
    np.testing.assert_allclose(
        concatenated.get_times(),
        np.arange(4000) / 30_000,
        rtol=0,
        atol=0.01 / 30_000,
    )
    for recording, traces, timestamps in members:
        np.testing.assert_array_equal(recording.get_traces(), traces)
        np.testing.assert_array_equal(recording.get_channel_gains(), [2, 2])
        np.testing.assert_array_equal(recording.get_channel_offsets(), [-7, -7])
        np.testing.assert_array_equal(
            recording.sample_index_to_time(np.arange(2000)), timestamps
        )


@pytest.mark.parametrize(
    "clocks", [("rate", "rate"), ("rate", "timestamps"), ("timestamps", "rate")]
)
@pytest.mark.integration
@pytest.mark.nwb
@pytest.mark.io_heavy
def test_concat_persisted_rate_and_timestamp_grids(tmp_path, clocks):
    """Known rates and inferred rates share the same acquisition grid."""
    members = [
        _persisted_concat_member(tmp_path, index, start, clock=clock)
        for index, (start, clock) in enumerate(zip((17, 0), clocks))
    ]
    recordings = [recording for recording, _, _ in members]
    assert [recording.has_time_vector() for recording in recordings] == [
        clock == "timestamps" for clock in clocks
    ]
    concatenated = build_concatenated_recording(recordings)
    np.testing.assert_array_equal(
        concatenated.get_traces(),
        np.concatenate([traces for _, traces, _ in members]),
    )
    np.testing.assert_array_equal(
        concatenated.get_traces(return_in_uV=True),
        np.concatenate([traces * 2 - 7 for _, traces, _ in members]),
    )
    np.testing.assert_allclose(
        concatenated.get_times(),
        np.arange(4000) / 30_000,
        rtol=0,
        atol=0.01 / 30_000,
    )


@pytest.mark.parametrize(
    "offsets, second_fs", [((0, 17), 30_000.3), ((-10_000, 0), 29_999.7)]
)
@pytest.mark.integration
@pytest.mark.nwb
@pytest.mark.io_heavy
def test_concat_persisted_timestamp_rates_reject_material_drift(
    tmp_path, offsets, second_fs
):
    """Both drift directions exceed roundoff, including a large negative anchor."""
    recordings = [
        _persisted_concat_member(tmp_path, 0, offsets[0])[0],
        _persisted_concat_member(tmp_path, 1, offsets[1], second_fs)[0],
    ]
    with pytest.raises(ValueError, match="sampling frequency"):
        build_concatenated_recording(recordings)


def _virtual_timestamp_recording(fs, n_samples):
    """A coherent long recording with a lazy clock; allocate only requested data."""
    import spikeinterface as si

    class LazyClock:
        ndim = 1

        def __getitem__(self, item):
            indices = (
                np.arange(*item.indices(n_samples))
                if isinstance(item, slice)
                else np.asarray(item)
            )
            return 10_000 + indices / fs

        def __array__(self, *args, **kwargs):
            raise AssertionError("The full clock must not be materialized")

    class Segment(si.BaseRecordingSegment):
        def __init__(self):
            super().__init__(time_vector=LazyClock())

        def get_num_samples(self):
            return n_samples

        def get_traces(self, start_frame, end_frame, channel_indices):
            n_channels = (
                2
                if channel_indices is None
                else len(np.arange(2)[channel_indices])
            )
            return np.zeros(
                (end_frame - start_frame, n_channels), dtype=np.float32
            )

    recording = si.BaseRecording(fs, [1, 2], "float32")
    recording.add_recording_segment(Segment())
    recording.set_channel_locations([[0, 0], [0, 20]])
    return recording


@pytest.mark.parametrize(
    "sample_counts, allowed",
    [
        ((150_000_000, 150_000_000), True),
        ((600_000_000, 150_000_000), False),
        ((150_000_000, 600_000_000), False),
    ],
    ids=["below_one_frame", "anchor_above_one_frame", "member_above_one_frame"],
)
@pytest.mark.unit
def test_timestamp_rate_roundoff_is_bounded_by_both_member_durations(
    sample_counts, allowed
):
    """The same rate uncertainty is safe below one frame and refused above it."""
    from spyglass.spikesorting.v2._concat_recording import (
        assert_concat_compatible,
    )

    recordings = [
        _virtual_timestamp_recording(fs, n_samples)
        for fs, n_samples in zip((30_000.0, 30_000.0001), sample_counts)
    ]
    # The supplied rates differ by 0.0001 Hz: 150 million frames accumulate
    # 0.5 frame of drift; 600 million accumulate 2 frames, independently of
    # the implementation's timestamp-precision calculation.
    if allowed:
        assert_concat_compatible(recordings)
    else:
        with pytest.raises(ValueError, match="sampling frequency"):
            assert_concat_compatible(recordings)


@pytest.mark.unit
def test_assert_concat_compatible_rejects_channel_id_mismatch():
    """A member with different channel ids (or count) is rejected early with a
    clear message instead of failing deep in SI's concatenate_recordings."""
    from spyglass.spikesorting.v2._concat_recording import (
        assert_concat_compatible,
    )

    locs = [[0.0, 0.0], [0.0, 20.0]]
    a = _rec_with_locations(100, [1, 2], locs)
    b = _rec_with_locations(50, [1, 3], locs)  # channel id 3 != 2
    with pytest.raises(ValueError, match="channel ids"):
        assert_concat_compatible([a, b])

    c = _rec_with_locations(50, [1], [[0.0, 0.0]])  # different count
    with pytest.raises(ValueError, match="channel ids"):
        assert_concat_compatible([a, c])


@pytest.mark.unit
def test_assert_concat_compatible_rejects_geometry_mismatch():
    """Members with matching channel ids but different probe geometry are
    rejected -- cross-session waveforms must align channel-for-channel. This is
    the case SI's id-only check would let through silently."""
    from spyglass.spikesorting.v2._concat_recording import (
        assert_concat_compatible,
    )

    a = _rec_with_locations(100, [1, 2], [[0.0, 0.0], [0.0, 20.0]])
    b = _rec_with_locations(50, [1, 2], [[0.0, 0.0], [0.0, 40.0]])
    with pytest.raises(ValueError, match="geometry"):
        assert_concat_compatible([a, b])


@pytest.mark.unit
def test_assert_concat_compatible_rejects_mismatched_fs():
    """Members with the same channel ids/geometry but different sampling
    frequencies cannot be stitched into one continuous timeline."""
    from spyglass.spikesorting.v2._concat_recording import (
        assert_concat_compatible,
    )

    locs = [[0.0, 0.0], [0.0, 20.0]]
    a = _rec_with_locations(100, [1, 2], locs, fs=30_000.0)
    b = _rec_with_locations(50, [1, 2], locs, fs=20_000.0)
    with pytest.raises(ValueError, match="sampling frequency"):
        assert_concat_compatible([a, b])

    # A sub-Hz drift that NumPy's default np.isclose (rtol=1e-5) would
    # silently accept must still be rejected (it mis-times every spike).
    near = _rec_with_locations(50, [1, 2], locs, fs=30_000.3)
    with pytest.raises(ValueError, match="sampling frequency"):
        assert_concat_compatible([a, near])

    # Supplied rates have no timestamp-inference uncertainty, even when the
    # difference is tiny enough to permit for an offset explicit clock.
    for fs in (30_000.000001, 29_999.999999):
        tiny = _rec_with_locations(50, [1, 2], locs, fs=fs)
        with pytest.raises(ValueError, match="sampling frequency"):
            assert_concat_compatible([a, tiny])


def _rec_with_dtype(n_samples, channel_ids, locations, dtype):
    """A synthetic NumpyRecording with an explicit sample dtype."""
    import spikeinterface as si

    rec = si.NumpyRecording(
        [np.zeros((n_samples, len(channel_ids)), dtype=dtype)],
        sampling_frequency=30_000.0,
        channel_ids=channel_ids,
    )
    rec.set_dummy_probe_from_locations(np.asarray(locations, dtype=float))
    return rec


@pytest.mark.unit
def test_assert_concat_compatible_rejects_mismatched_dtype_gain():
    """Members with mismatched sample dtype, channel gains, or channel offsets
    are rejected -- concatenation would silently combine differently-scaled
    traces into one recording."""
    from spyglass.spikesorting.v2._concat_recording import (
        assert_concat_compatible,
    )

    locs = [[0.0, 0.0], [0.0, 20.0]]

    # dtype mismatch.
    a = _rec_with_dtype(100, [1, 2], locs, np.float32)
    b_int = _rec_with_dtype(50, [1, 2], locs, np.int16)
    with pytest.raises(ValueError, match="dtype"):
        assert_concat_compatible([a, b_int])

    # gain mismatch.
    a_gain = _rec_with_locations(100, [1, 2], locs)
    a_gain.set_channel_gains([1.0, 1.0])
    b_gain = _rec_with_locations(50, [1, 2], locs)
    b_gain.set_channel_gains([2.0, 2.0])
    with pytest.raises(ValueError, match="gain"):
        assert_concat_compatible([a_gain, b_gain])

    # offset mismatch.
    a_off = _rec_with_locations(100, [1, 2], locs)
    a_off.set_channel_offsets([0.0, 0.0])
    b_off = _rec_with_locations(50, [1, 2], locs)
    b_off.set_channel_offsets([5.0, 5.0])
    with pytest.raises(ValueError, match="offset"):
        assert_concat_compatible([a_off, b_off])


# ---------- electrode_signature_from_rows ----------------------------------


def test_assert_concat_compatible_accepts_absent_optional_metadata():
    import spikeinterface as si

    from spyglass.spikesorting.v2._concat_recording import (
        assert_concat_compatible,
    )

    recordings = [
        si.NumpyRecording(np.zeros((n, 2), dtype=np.float32), 30_000)
        for n in (100, 50)
    ]
    assert all(not rec.has_channel_location() for rec in recordings)
    assert all(rec.get_channel_gains() is None for rec in recordings)
    assert all(rec.get_channel_offsets() is None for rec in recordings)
    assert_concat_compatible(recordings)


def test_assert_concat_compatible_rejects_optional_geometry_presence_mismatch():
    import spikeinterface as si

    from spyglass.spikesorting.v2._concat_recording import (
        assert_concat_compatible,
    )

    present = _rec_with_locations(100, [0, 1], [[0, 0], [0, 20]])
    absent = si.NumpyRecording(np.zeros((50, 2), dtype=np.float32), 30_000)
    with pytest.raises(ValueError, match="geometry presence differs"):
        assert_concat_compatible([present, absent])


@pytest.mark.parametrize(
    "getter",
    ["get_channel_locations", "get_channel_gains", "get_channel_offsets"],
)
@pytest.mark.parametrize("error_type", [OSError, ValueError])
@pytest.mark.parametrize("member", [0, 1])
def test_assert_concat_compatible_propagates_metadata_read_errors(
    monkeypatch, getter, error_type, member
):
    from spyglass.spikesorting.v2._concat_recording import (
        assert_concat_compatible,
    )

    recordings = [
        _rec_with_locations(n, [0, 1], [[0, 0], [0, 20]]) for n in (100, 50)
    ]
    failure = error_type("metadata unreadable")

    def broken():
        raise failure

    monkeypatch.setattr(recordings[member], getter, broken)
    with pytest.raises(error_type, match="metadata unreadable") as raised:
        assert_concat_compatible(recordings)
    assert raised.value is failure


@pytest.mark.unit
def test_electrode_signature_distinguishes_reused_ids_across_groups():
    """Two members whose sort groups carry the SAME electrode ids and regions
    but on DIFFERENT electrode groups (ids reused across probes -- a documented
    hazard, since the Electrode PK is (nwb, electrode_group_name, electrode_id))
    must get DIFFERENT signatures. Dropping the group name would collapse two
    physically distinct electrode spaces into one and let the concat read one
    member in the other's frame."""
    from spyglass.spikesorting.v2._concat_recording import (
        electrode_signature_from_rows,
    )

    rows_a = [
        {"electrode_group_name": "probeA", "electrode_id": 0},
        {"electrode_group_name": "probeA", "electrode_id": 1},
    ]
    rows_b = [
        {"electrode_group_name": "probeB", "electrode_id": 0},
        {"electrode_group_name": "probeB", "electrode_id": 1},
    ]
    region_a = {("probeA", 0): "ca1", ("probeA", 1): "ca1"}
    region_b = {("probeB", 0): "ca1", ("probeB", 1): "ca1"}

    sig_a = electrode_signature_from_rows(rows_a, region_a)
    sig_b = electrode_signature_from_rows(rows_b, region_b)

    assert sig_a != sig_b


@pytest.mark.unit
def test_electrode_signature_matches_for_identical_physical_electrodes():
    """Identical electrode group / id / region across members -> equal
    signature, regardless of fetched-row order (signature is order-invariant).
    """
    from spyglass.spikesorting.v2._concat_recording import (
        electrode_signature_from_rows,
    )

    rows = [
        {"electrode_group_name": "probeA", "electrode_id": 1},
        {"electrode_group_name": "probeA", "electrode_id": 0},
    ]
    region = {("probeA", 0): "ca1", ("probeA", 1): "ca1"}

    assert electrode_signature_from_rows(
        rows, region
    ) == electrode_signature_from_rows(list(reversed(rows)), region)


@pytest.mark.unit
def test_electrode_signature_marks_missing_region_as_none():
    """An electrode absent from the region map maps to None (best-effort
    region), not a KeyError."""
    from spyglass.spikesorting.v2._concat_recording import (
        electrode_signature_from_rows,
    )

    sig = electrode_signature_from_rows(
        [{"electrode_group_name": "probeA", "electrode_id": 0}], {}
    )
    assert sig == (("probeA", 0, None),)
