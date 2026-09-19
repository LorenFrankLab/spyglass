"""Plane selection and uniqueness checks for raw electrode geometry.

Real Frank-lab tetrode files store contacts in the x-z plane (``rel_y`` is 0
for every contact, ``rel_z`` is +-6.25 um). SpikeInterface projects 3D channel
locations onto x-y whenever it builds a probe
(``create_dummy_probe_from_locations(axes="xy")``), which collapses such a
tetrode onto two coincident positions and fails at analyzer build with
"Contact positions must be unique within a probe".

These are DB-free unit tests of the three helpers that pick the plane keeping
every contact distinct, write it back as 2D channel locations, and assert the
effective geometry is usable.
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.spikesorting.v2._recording_geometry import (
    assert_unique_contact_positions,
    normalize_channel_locations,
    select_distinct_plane,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def xz_tetrode_locations():
    """A real-file tetrode: four contacts in the x-z plane, ``rel_y`` all 0.

    The x-y projection SpikeInterface would take has only two distinct
    positions, so this array discriminates a plane chooser from a no-op.
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
    """A planar x-y tetrode square (``rel_z`` all 0) -- x-y must win."""
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

    ``si.generate_recording`` attaches a dummy linear probe, so its
    ``contact_vector`` property shadows ``location`` in
    ``get_channel_locations`` and blocks ``set_channel_locations``. The helpers
    under test run on the recording read from the NWB file, which carries
    locations only (``NwbRecordingExtractor`` calls ``set_channel_locations``),
    so the probe is removed here to match.
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

    ``sort_group_geometry_problem`` maps an all-unpositioned group onto the
    all-zero legacy geometry so the ``tetrode_12.5`` repair can rescue it, and
    lets the run start. Raising here would fail that same group at make, after
    preflight cleared it. Partial non-finite rows still raise (above): there
    the repair does not apply and the missing contacts are a real defect.
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
    """A probe already makes the geometry read-only -- fail, don't skip."""
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

    ``select_sort_group_channels`` keeps a ``specific`` reference channel
    through plane selection and ``apply_spatial_preprocessing`` drops it
    afterwards, so a group can reach this point with its 3D locations intact
    and an x-y projection that happens to be distinct. The writer refuses that
    recording -- but only after the whole compute has run.
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

    # The analyzer path deliberately loads 3D artifacts (the writer persists
    # rel_z, so ``NwbRecordingExtractor`` rebuilds a 3D ``location``) and
    # projects them with ``probe.to_2d()``; it opts out of the 2D requirement.
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

    ``maybe_apply_tetrode_geometry`` needs a recording and logs its verdict;
    preflight needs the same verdict from probe metadata alone. Both now ask
    this predicate, so its four gates are pinned here: one all-true case, and
    one case per gate that flips it to False with everything else held.
    """
    from spyglass.spikesorting.v2._recording_geometry import (
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
    # Gate 4: four tetrode channels, but drawn from two electrode groups --
    # they are not one physical tetrode.
    assert tetrode_repair_applies(tetrode, ("0", "0", "1", "1"), 4) is False
    # An empty group must answer False, not raise on the empty probe set.
    assert tetrode_repair_applies((), (), 0) is False
