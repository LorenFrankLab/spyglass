"""DB-free unit tests for the UnitAnnotation spike-selection helper.

``UnitAnnotation.fetch_unit_spikes`` selects spike-time arrays by the NWB units
table's TRUE unit id (a dict keyed by real ids), not by a positional index into
the spike-times list -- positional indexing is wrong for the sparse /
non-contiguous id sets that v2 and merge-applied v0/v1 curations produce. The
per-id validation is extracted here so the "requested id is not a true id" path
(an old positional annotation on a sparse curation) raises an actionable error
rather than a bare ``KeyError``, and is pinned without the (JAX-skipped) v1
integration fixture.
"""

import pytest

from spyglass.spikesorting.analysis.v1._unit_annotation_helpers import (
    spikes_for_requested_units,
)

pytestmark = pytest.mark.unit


def test_selects_by_true_id_not_position():
    """Spikes are selected by the NWB true unit id; a sparse id set {2,4} is
    resolved by value, so requesting 4 returns the 4-keyed spikes (a positional
    index would return the wrong array)."""
    mapping = {2: "spikes_2", 4: "spikes_4"}
    assert spikes_for_requested_units(mapping, [4, 2], merge_id="m") == [
        "spikes_4",
        "spikes_2",
    ]


def test_raises_actionable_error_on_missing_id():
    """A requested id absent from the true-id set raises an actionable
    ValueError -- naming the valid ids and the positional->true-id change --
    not a bare KeyError."""
    mapping = {2: "spikes_2", 4: "spikes_4"}
    with pytest.raises(ValueError, match="not a unit id"):
        spikes_for_requested_units(mapping, [3], merge_id="abc")
    with pytest.raises(ValueError, match=r"valid unit ids.*\[2, 4\]"):
        spikes_for_requested_units(mapping, [0], merge_id="abc")
    with pytest.raises(ValueError, match="positional"):
        spikes_for_requested_units(mapping, [0], merge_id="abc")


def test_numpy_and_python_ids_interchangeable():
    """A numpy requested id matches the plain-int key (and vice versa), so a
    fetched-then-requested id is not spuriously reported missing."""
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
