"""Plane selection and uniqueness checks for raw electrode geometry.

Frank-lab tetrode files store contacts in the x-z plane (``rel_y`` is 0).
SpikeInterface projects 3D locations onto x-y when it builds a probe, which
collapses such a tetrode onto two positions. No DB.
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.spikesorting.v2._recording.geometry import (
    assert_unique_contact_positions,
    classify_missing_geometry,
    normalize_channel_locations,
    select_distinct_plane,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def xz_tetrode_locations():
    """A real-file tetrode: four contacts in the x-z plane, ``rel_y`` all 0.

    Its x-y projection has only two distinct positions.
    """
    return np.array(
        [
            [6.25, 0.0, 6.25],
            [-6.25, 0.0, 6.25],
            [-6.25, 0.0, -6.25],
            [6.25, 0.0, -6.25],
        ]
    )


@pytest.fixture
def xy_square_locations():
    """A planar x-y tetrode square (``rel_z`` all 0); x-y must win."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 12.5, 0.0],
            [12.5, 0.0, 0.0],
            [12.5, 12.5, 0.0],
        ]
    )


@pytest.fixture
def yz_plane_locations():
    """Contacts separated only in y-z: ``rel_x`` constant, both other pairs
    collapse a pair of contacts."""
    return np.array(
        [
            [0.0, 6.25, 6.25],
            [0.0, -6.25, 6.25],
            [0.0, -6.25, -6.25],
            [0.0, 6.25, -6.25],
        ]
    )


@pytest.fixture
def nan_locations():
    """A NULL ``rel_z`` column read back as NaN on two contacts."""
    return np.array(
        [
            [0.0, 0.0, np.nan],
            [0.0, 12.5, 0.0],
            [12.5, 0.0, np.nan],
            [12.5, 12.5, 0.0],
        ]
    )


@pytest.fixture
def all_zero_locations():
    """Legacy geometry that was never written: every coordinate 0."""
    return np.zeros((4, 3))


def _probeless_recording(num_channels: int):
    """Build a synthetic recording with no probe attached.

    Recordings read from NWB carry locations only; a probe would shadow
    ``location`` and block ``set_channel_locations``.
    """
    import spikeinterface as si

    recording = si.generate_recording(
        num_channels=num_channels, durations=[0.1]
    )
    recording.delete_property("contact_vector")
    return recording


def test_select_distinct_plane_prefers_xy_then_xz(
    xz_tetrode_locations,
    xy_square_locations,
    yz_plane_locations,
    all_zero_locations,
):
    """x-y when it separates the contacts, else the first pair that does."""
    axes, positions = select_distinct_plane(xz_tetrode_locations)
    assert axes == "xz", "y is constant, so x-y cannot separate the contacts"
    assert positions.shape == (4, 2)
    np.testing.assert_array_equal(positions, xz_tetrode_locations[:, [0, 2]])
    assert len(np.unique(positions, axis=0)) == 4

    axes, positions = select_distinct_plane(xy_square_locations)
    assert axes == "xy", "a planar x-y probe must not be re-projected"
    np.testing.assert_array_equal(positions, xy_square_locations[:, [0, 1]])

    axes, positions = select_distinct_plane(yz_plane_locations)
    assert axes == "yz", "x is constant, so only y-z separates the contacts"
    np.testing.assert_array_equal(positions, yz_plane_locations[:, [1, 2]])
    assert len(np.unique(positions, axis=0)) == 4

    assert (
        select_distinct_plane(all_zero_locations) is None
    ), "no plane separates coincident contacts; the caller must raise"

    axes, positions = select_distinct_plane(np.array([[1.0, 2.0, 3.0]]))
    assert axes == "xy", "a single contact is distinct in every plane"
    np.testing.assert_array_equal(positions, [[1.0, 2.0]])


def test_classify_missing_geometry(
    xz_tetrode_locations, nan_locations, all_zero_locations
):
    """One predicate decides how much of a group's geometry is missing.

    Shared by preflight and the recording stage so they agree.
    """
    assert classify_missing_geometry(xz_tetrode_locations) == "none"
    assert classify_missing_geometry(all_zero_locations) == "none"
    assert classify_missing_geometry(np.full((4, 3), np.nan)) == "complete"

    # A whole row missing, and a single missing coordinate on otherwise
    # positioned rows, are the same verdict: some contacts cannot be placed.
    partial_row = np.array(xz_tetrode_locations)
    partial_row[2] = np.nan
    assert classify_missing_geometry(partial_row) == "partial"
    assert classify_missing_geometry(nan_locations) == "partial"

    # Finite x and y, NaN z: every row is incomplete, but the group is not
    # unpositioned.
    nan_z = np.array(xz_tetrode_locations)
    nan_z[:, 2] = np.nan
    assert classify_missing_geometry(nan_z) == "partial"

    # np.inf is not a position either.
    infinite = np.array(xz_tetrode_locations)
    infinite[1, 0] = np.inf
    assert classify_missing_geometry(infinite) == "partial"

    # An empty set has no verdict: reading it as "wholly unpositioned" would
    # route a group with no contacts onto the legacy all-zero geometry.
    with pytest.raises(ValueError, match="no contacts"):
        classify_missing_geometry(np.empty((0, 3)))


def test_normalize_rejects_a_wholly_non_finite_coordinate(
    xz_tetrode_locations,
):
    """Finite x/y with a NaN z is a defect, not an unpositioned group.

    Only a set with no finite coordinate is repaired as a tetrode.
    """
    recording = _probeless_recording(4)
    nan_z = np.array(xz_tetrode_locations)
    nan_z[:, 2] = np.nan
    recording.set_channel_locations(nan_z)

    with pytest.raises(ValueError, match="finite"):
        normalize_channel_locations(recording)


def test_select_distinct_plane_rejects_non_finite(nan_locations):
    """A NULL ``rel_*`` column must not masquerade as a distinct contact."""
    with pytest.raises(ValueError) as excinfo:
        select_distinct_plane(nan_locations)
    message = str(excinfo.value)
    assert "finite" in message
    assert "[0, 2]" in message, "the message must name the offending rows"
    assert "Probe.Electrode" in message


def test_normalize_rejects_non_finite_locations(nan_locations):
    """The raise propagates through the recording-level entry point."""
    recording = _probeless_recording(4)
    recording.set_channel_locations(nan_locations)

    with pytest.raises(ValueError, match="finite"):
        normalize_channel_locations(recording)


def test_normalize_treats_all_non_finite_locations_as_absent():
    """A group with NO usable coordinates is passed through, not rejected.

    Preflight passes such a group to the ``tetrode_12.5`` repair. Partial
    non-finite rows still raise.
    """
    recording = _probeless_recording(4)
    all_nan = np.full((4, 3), np.nan)
    recording.set_channel_locations(all_nan)

    result = normalize_channel_locations(recording)

    locations = np.asarray(result.get_property("location"), dtype=float)
    assert locations.shape == (4, 3), "unusable geometry must be left alone"
    assert np.isnan(locations).all()


def test_normalize_on_channel_subset(xz_tetrode_locations):
    """Normalization runs on the sliced group, not the parent's channels."""
    recording = _probeless_recording(4)
    recording.set_channel_locations(xz_tetrode_locations)
    subset = recording.select_channels(recording.get_channel_ids()[:3])

    result = normalize_channel_locations(subset)

    locations = result.get_channel_locations()
    assert locations.shape == (3, 2)
    np.testing.assert_array_equal(
        locations, xz_tetrode_locations[:3][:, [0, 2]]
    )
    assert (
        len(np.unique(locations, axis=0)) == 3
    ), "the three retained contacts must stay distinct"
    assert (
        result.get_property("contact_vector") is None
    ), "normalization must not build a probe"


def test_normalize_uses_only_the_retained_contacts(xz_tetrode_locations):
    """The plane is chosen from the contacts that stay in the sort surface.

    ``rel_*`` are per probe type, so a ``specific`` reference on a second
    identical probe can share a member's coordinates. The four retained rows
    separate in x-z; the five sliced rows do not.
    """
    locations = np.vstack([xz_tetrode_locations, xz_tetrode_locations[:1]])

    unrestricted = _probeless_recording(5)
    unrestricted.set_channel_locations(locations)
    assert normalize_channel_locations(unrestricted).has_3d_locations(), (
        "fixture must be unseparable WITH the reference, or this test proves "
        "nothing"
    )

    recording = _probeless_recording(5)
    recording.set_channel_locations(locations)
    channel_ids = list(recording.get_channel_ids())

    result = normalize_channel_locations(recording, channel_ids=channel_ids[:4])

    projected = np.asarray(result.get_property("location"), dtype=float)
    assert projected.shape == (5, 2), (
        "the chosen projection must be applied to EVERY channel, including "
        "the reference that is still on the surface"
    )
    np.testing.assert_array_equal(projected, locations[:, [0, 2]])
    assert (
        len(np.unique(projected[:4], axis=0)) == 4
    ), "the four retained contacts must be distinct in the chosen plane"

    # After the reference is dropped, the rest passes the recording gate.
    assert_unique_contact_positions(result.remove_channels(channel_ids[4:]))


def test_normalize_still_requires_every_sliced_contact_to_be_placeable(
    xz_tetrode_locations,
):
    """Only the PLANE is scoped to the retained contacts, not finiteness.

    ``interpolate`` builds kriging weights from every channel, so a NULL
    ``rel_*`` on the reference would spread NaN.
    """
    locations = np.vstack([xz_tetrode_locations, [[np.nan, np.nan, np.nan]]])
    recording = _probeless_recording(5)
    recording.set_channel_locations(locations)
    channel_ids = list(recording.get_channel_ids())

    with pytest.raises(ValueError, match="finite"):
        normalize_channel_locations(recording, channel_ids=channel_ids[:4])


def test_normalize_screens_a_positioned_reference_beside_unpositioned_members():
    """Unpositioned members plus a positioned reference is still a defect.

    The ``tetrode_12.5`` pass-through applies only when the whole sliced set
    has no geometry.
    """
    locations = np.full((5, 3), np.nan)
    locations[4] = [10.0, 20.0, 30.0]
    recording = _probeless_recording(5)
    recording.set_channel_locations(locations)
    channel_ids = list(recording.get_channel_ids())

    with pytest.raises(ValueError, match="finite"):
        normalize_channel_locations(recording, channel_ids=channel_ids[:4])


def test_normalize_reduces_planar_geometry_to_xy(xy_square_locations):
    """A 3D but already-planar probe is reduced to its x-y columns."""
    recording = _probeless_recording(4)
    recording.set_channel_locations(xy_square_locations)

    result = normalize_channel_locations(recording)

    locations = result.get_property("location")
    assert locations.shape == (4, 2), "3D locations must be written back as 2D"
    np.testing.assert_array_equal(locations, xy_square_locations[:, [0, 1]])


def test_normalize_leaves_2d_and_missing_locations_alone():
    """2D locations and absent locations are both returned untouched."""
    planar = np.array([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0], [0.0, 60.0]])
    recording = _probeless_recording(4)
    recording.set_channel_locations(planar)

    result = normalize_channel_locations(recording)

    np.testing.assert_array_equal(result.get_channel_locations(), planar)

    without_locations = _probeless_recording(4)
    without_locations.delete_property("location")

    result = normalize_channel_locations(without_locations)

    assert result.get_property("location") is None


def test_normalize_refuses_when_probe_attached(xz_tetrode_locations):
    """A probe makes the geometry read-only, so this raises, not skips."""
    recording = _probeless_recording(4)
    recording.set_channel_locations(xz_tetrode_locations)
    recording.set_dummy_probe_from_locations(xz_tetrode_locations, axes="xz")

    with pytest.raises(ValueError, match="probe"):
        normalize_channel_locations(recording)


def test_assert_unique_contact_positions_requires_all_distinct():
    """Coincident 2D contacts raise and name the table to fix."""
    duplicated = np.array([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0], [0.0, 40.0]])
    recording = _probeless_recording(4)
    recording.set_channel_locations(duplicated)

    with pytest.raises(ValueError) as excinfo:
        assert_unique_contact_positions(recording)
    message = str(excinfo.value)
    assert "Probe.Electrode" in message
    assert "40.0" in message, "the message must name the offending positions"

    distinct = np.array([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0], [0.0, 60.0]])
    recording = _probeless_recording(4)
    recording.set_channel_locations(distinct)
    assert assert_unique_contact_positions(recording) is None

    single = _probeless_recording(1)
    single.set_channel_locations(np.array([[0.0, 0.0]]))
    assert assert_unique_contact_positions(single) is None


def test_assert_unique_contact_positions_refuses_3d_locations():
    """Still-3D locations fail at the gate, not later inside the writer.

    A group can arrive 3D with a distinct x-y projection; the writer would
    refuse it only after the compute.
    """
    recording = _probeless_recording(4)
    # x-y alone separates these four, so the coincidence check would pass.
    recording.set_channel_locations(
        np.array(
            [
                [0.0, 0.0, 6.25],
                [0.0, 12.5, 6.25],
                [12.5, 0.0, -6.25],
                [12.5, 12.5, -6.25],
            ]
        )
    )

    with pytest.raises(ValueError) as excinfo:
        assert_unique_contact_positions(recording)
    message = str(excinfo.value)
    assert "2D plane" in message
    assert "Probe.Electrode" in message

    # The analyzer path loads 3D artifacts and projects them itself.
    assert assert_unique_contact_positions(recording, require_2d=False) is None


def test_assert_unique_contact_positions_without_any_geometry():
    """No geometry at all is an actionable error, not SI's bare Exception."""
    recording = _probeless_recording(4)
    recording.delete_property("location")

    with pytest.raises(ValueError) as excinfo:
        assert_unique_contact_positions(recording)
    assert "Probe.Electrode" in str(excinfo.value)


def test_tetrode_repair_applies_gates():
    """Each gate of the legacy ``tetrode_12.5`` repair, one at a time.

    Shared by ``maybe_apply_tetrode_geometry`` and preflight.
    """
    from spyglass.spikesorting.v2._recording.geometry import (
        tetrode_repair_applies,
    )

    tetrode = ("tetrode_12.5",) * 4
    one_group = ("0",) * 4

    assert tetrode_repair_applies(tetrode, one_group, 4) is True

    # Gate 1: the group spans more than one probe type.
    assert (
        tetrode_repair_applies(
            ("tetrode_12.5", "tetrode_12.5", "tetrode_12.5", "other_probe"),
            one_group,
            4,
        )
        is False
    )
    # Gate 2: a single probe, but not the one the 12.5 um square describes.
    assert (
        tetrode_repair_applies(
            ("128c-4s6mm6cm-15um-26um-sl",) * 4, one_group, 4
        )
        is False
    )
    # Gate 3: not exactly four channels (the square has four corners).
    assert tetrode_repair_applies(tetrode, one_group, 3) is False
    assert tetrode_repair_applies(("tetrode_12.5",) * 5, ("0",) * 5, 5) is False
    # Gate 4: four tetrode channels from two electrode groups.
    assert tetrode_repair_applies(tetrode, ("0", "0", "1", "1"), 4) is False
    # An empty group must answer False, not raise on the empty probe set.
    assert tetrode_repair_applies((), (), 0) is False


@pytest.mark.parametrize(
    "case, verdict",
    [
        ("column", "plane"),
        ("unpositioned", "collapsed"),
        ("coincident", "collapsed"),
        ("one_null", "partial"),
    ],
)
def test_effective_contact_plane(case, verdict):
    """Preflight's shared reading of registered positions: a separable set
    gives its plane, a set with no position reads as all-zero (collapsed),
    coincident contacts collapse, and a partly missing set is ``partial``
    with its positions untouched."""
    from spyglass.spikesorting.v2._recording.geometry import (
        effective_contact_plane,
    )

    locations = np.column_stack([np.zeros(4), np.arange(4) * 20.0, np.zeros(4)])
    if case == "unpositioned":
        locations = np.full((4, 3), np.nan)
    elif case == "coincident":
        locations[1] = locations[0]
    elif case == "one_null":
        locations[2, 2] = np.nan
    got, positions, planar = effective_contact_plane(locations)
    assert got == verdict
    if verdict == "plane":
        np.testing.assert_array_equal(planar, locations[:, :2])
    else:
        assert planar is None
    if case == "unpositioned":
        np.testing.assert_array_equal(positions, np.zeros((4, 3)))
    if case == "one_null":
        np.testing.assert_array_equal(positions, locations)


def test_sort_group_planar_positions_matches_the_recording_stage():
    """The plotted positions are the ones the recording stage installs.

    A single-group ``tetrode_12.5`` tetrode always gets the 12.5 um square,
    whatever is registered; other groups use the plane that separates every
    contact (a Frank-lab tetrode lies in x-z); a group no plane separates
    and no repair rescues has none.
    """
    from spyglass.spikesorting.v2._recording.geometry import (
        sort_group_planar_positions,
    )

    xz_tetrode = np.array(
        [[0.0, 5.0, 0.0], [0.0, 5.0, 20.0], [20.0, 5.0, 0.0], [20.0, 5.0, 20.0]]
    )
    source, planar = sort_group_planar_positions(
        xz_tetrode, ("tetrode_12.5",) * 4, ("0",) * 4
    )
    assert source == "tetrode_12.5"
    np.testing.assert_array_equal(
        planar, [[0, 0], [0, 12.5], [12.5, 0], [12.5, 12.5]]
    )

    source, planar = sort_group_planar_positions(
        xz_tetrode, ("other_probe",) * 4, ("0",) * 4
    )
    assert source == "xz"
    np.testing.assert_array_equal(planar, xz_tetrode[:, [0, 2]])

    assert sort_group_planar_positions(
        np.zeros((4, 3)), ("other_probe",) * 4, ("0",) * 4
    ) == (None, None)
