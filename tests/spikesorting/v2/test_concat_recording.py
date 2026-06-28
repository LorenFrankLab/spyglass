"""Unit tests for the DB-free concatenated-recording service helpers.

Drive ``_concat_recording`` directly -- motion-preset resolution and the
sample-boundary / spike-train back-mapping math are pure, and the
concatenate path runs on synthetic ``NumpyRecording`` objects with no DB.
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.spikesorting.v2._concat_recording import (
    AUTO_SAME_DAY_PRESET,
    build_concatenated_recording,
    cumulative_member_boundaries,
    member_split_key,
    resolve_motion_correction,
    split_unit_spike_trains,
)

pytestmark = pytest.mark.unit


# ---------- resolve_motion_correction --------------------------------------


def test_resolve_motion_none_skips():
    """``preset='none'`` resolves to no correction (None) with empty kwargs."""
    assert resolve_motion_correction(
        {"preset": "none", "preset_kwargs": {}}, is_multi_day=False
    ) == (None, {})


def test_resolve_motion_auto_same_day_maps_to_rigid_fast():
    """``preset='auto'`` maps to the same-day preset for a single-day group."""
    preset, kwargs = resolve_motion_correction(
        {"preset": "auto", "preset_kwargs": {}}, is_multi_day=False
    )
    assert preset == AUTO_SAME_DAY_PRESET == "rigid_fast"
    assert kwargs == {}


def test_resolve_motion_auto_multi_day_raises():
    """``preset='auto'`` is rejected for a multi-day group."""
    with pytest.raises(ValueError, match="single-day only"):
        resolve_motion_correction(
            {"preset": "auto", "preset_kwargs": {}}, is_multi_day=True
        )


def test_resolve_motion_explicit_preset_passes_through():
    """An explicit SI preset (and its kwargs) is returned unchanged, even on a
    multi-day group."""
    preset, kwargs = resolve_motion_correction(
        {"preset": "dredge_fast", "preset_kwargs": {"detect_kwargs": {"x": 1}}},
        is_multi_day=True,
    )
    assert preset == "dredge_fast"
    assert kwargs == {"detect_kwargs": {"x": 1}}


def test_resolve_motion_missing_preset_raises():
    """A blob with no ``preset`` key is rejected."""
    with pytest.raises(ValueError, match="no 'preset'"):
        resolve_motion_correction({}, is_multi_day=False)


# ---------- cumulative_member_boundaries -----------------------------------


def test_cumulative_boundaries_basic():
    """Boundaries are running totals; the last equals the grand total."""
    assert cumulative_member_boundaries([100, 50, 30]) == [100, 150, 180]


def test_cumulative_boundaries_empty():
    """No members -> no boundaries."""
    assert cumulative_member_boundaries([]) == []


# ---------- split_unit_spike_trains ----------------------------------------


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


def test_split_preserves_unit_ids_with_empty_arrays():
    """A unit absent from a member's span still appears, with an empty array."""
    trains = {3: np.array([5]), 4: np.array([150])}
    per_member = split_unit_spike_trains(trains, [100, 200])
    assert set(per_member[0]) == {3, 4}
    assert set(per_member[1]) == {3, 4}
    assert per_member[0][4].size == 0
    assert per_member[1][3].size == 0


# ---------- build_concatenated_recording -----------------------------------


def test_build_concatenated_recording_no_motion_sums_samples():
    """With ``motion_preset=None`` the members are stitched into one segment
    whose sample count is the sum and whose channels are preserved."""
    import spikeinterface as si

    fs = 30_000.0
    rng = np.random.default_rng(0)
    rec_a = si.NumpyRecording(
        [rng.normal(0, 1, size=(300, 4)).astype(np.float32)],
        sampling_frequency=fs,
    )
    rec_b = si.NumpyRecording(
        [rng.normal(0, 1, size=(200, 4)).astype(np.float32)],
        sampling_frequency=fs,
    )
    concat = build_concatenated_recording(
        [rec_a, rec_b], motion_preset=None
    )
    assert concat.get_num_segments() == 1
    assert concat.get_num_samples() == 500
    assert list(concat.get_channel_ids()) == list(rec_a.get_channel_ids())


def test_build_concatenated_recording_merges_overlapping_motion_kwargs(
    monkeypatch,
):
    """An overlapping key (e.g. n_jobs) in both preset_kwargs and job_kwargs is
    merged into one correct_motion call (job kwargs win) instead of
    double-splatting into a TypeError."""
    import spikeinterface as si
    import spikeinterface.preprocessing as sp

    captured = {}

    def fake_correct_motion(recording, **kwargs):
        captured.update(kwargs)
        return recording

    # build_concatenated_recording does `from spikeinterface.preprocessing
    # import correct_motion` at call time, so patching the module attr is seen.
    monkeypatch.setattr(sp, "correct_motion", fake_correct_motion)
    rec = si.NumpyRecording(
        [np.zeros((100, 4), dtype=np.float32)], sampling_frequency=30_000.0
    )
    build_concatenated_recording(
        [rec],
        motion_preset="rigid_fast",
        preset_kwargs={"n_jobs": 1, "detect_kwargs": {"x": 1}},
        job_kwargs={"n_jobs": 4},
    )
    assert captured["preset"] == "rigid_fast"
    assert captured["n_jobs"] == 4  # resolved job kwargs win on conflict
    assert captured["detect_kwargs"] == {"x": 1}
    assert captured["output_motion"] is False
    assert captured["output_motion_info"] is False


@pytest.mark.parametrize(
    "bad_key, bad_val",
    [
        ("folder", "/tmp/motion"),  # side-artifact write
        ("overwrite", True),  # side-artifact / return-type contract
        ("output_motion", True),  # changes return type
        ("detect_kwargs", {"x": 1}),  # motion param outside the concat identity
    ],
)
def test_build_concatenated_recording_rejects_non_job_motion_job_kwargs(
    bad_key, bad_val
):
    """A non-SI-job key in the resolved motion job_kwargs is rejected before it
    can bind a correct_motion top-level param (side artifacts / return type /
    motion params) and bypass the concat persistence contract."""
    import spikeinterface as si

    rec = si.NumpyRecording(
        [np.zeros((50, 4), dtype=np.float32)], sampling_frequency=30_000.0
    )
    with pytest.raises(ValueError, match="non-job key"):
        build_concatenated_recording(
            [rec], motion_preset="rigid_fast", job_kwargs={bad_key: bad_val}
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


def test_assert_concat_compatible_accepts_matching_members():
    """Members sharing channel ids and geometry pass the pre-concat check."""
    from spyglass.spikesorting.v2._concat_recording import (
        assert_concat_compatible,
    )

    locs = [[0.0, 0.0], [0.0, 20.0]]
    a = _rec_with_locations(100, [1, 2], locs)
    b = _rec_with_locations(50, [1, 2], locs)
    assert_concat_compatible([a, b])  # no raise


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

    # The regression case: a sub-Hz drift NumPy's default np.isclose (rtol=1e-5)
    # would silently accept must still be rejected (it mis-times every spike).
    near = _rec_with_locations(50, [1, 2], locs, fs=30_000.3)
    with pytest.raises(ValueError, match="sampling frequency"):
        assert_concat_compatible([a, near])


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
