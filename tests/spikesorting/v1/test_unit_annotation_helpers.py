"""DB-free unit tests for the UnitAnnotation spike-selection helper.

``fetch_unit_spikes`` selects spikes by true NWB unit id, not position, so
sparse id sets (merge-applied curations) resolve correctly.
"""

import pytest

from spyglass.spikesorting.analysis.v1._unit_annotation_helpers import (
    spikes_for_requested_units,
)

pytestmark = pytest.mark.unit


def test_selects_by_true_id_not_position():
    """A sparse id set {2, 4} is resolved by value, not position."""
    mapping = {2: "spikes_2", 4: "spikes_4"}
    assert spikes_for_requested_units(mapping, [4, 2], merge_id="m") == [
        "spikes_4",
        "spikes_2",
    ]


def test_raises_actionable_error_on_missing_id():
    """A missing id raises a ValueError naming the valid ids."""
    mapping = {2: "spikes_2", 4: "spikes_4"}
    with pytest.raises(ValueError, match="not a unit id"):
        spikes_for_requested_units(mapping, [3], merge_id="abc")
    with pytest.raises(ValueError, match=r"valid unit ids.*\[2, 4\]"):
        spikes_for_requested_units(mapping, [0], merge_id="abc")
    with pytest.raises(ValueError, match="positional"):
        spikes_for_requested_units(mapping, [0], merge_id="abc")


def test_numpy_and_python_ids_interchangeable():
    """A numpy id matches the plain-int key, and vice versa."""
    np = pytest.importorskip("numpy")
    assert spikes_for_requested_units(
        {np.int64(2): "s"}, [2], merge_id="m"
    ) == ["s"]
    assert spikes_for_requested_units(
        {2: "s"}, [np.int64(2)], merge_id="m"
    ) == ["s"]


def test_empty_request_returns_empty():
    """No requested ids -> no spikes (and no error)."""
    assert spikes_for_requested_units({2: "s"}, [], merge_id="m") == []
