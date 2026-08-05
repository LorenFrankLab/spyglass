"""#1623: get_group_by_shank crashed on non-numeric electrode_group_name."""

import numpy as np
import pytest


def _electrode_array(group_names, n_per_group=4):
    """Build the minimal Electrode fetch() result get_group_by_shank reads.

    One shank per group and a shared reference electrode, so each electrode
    group yields exactly one sort group and sort_group_id encodes the order
    in which the group names were visited.
    """
    dtype = [
        ("electrode_id", "i8"),
        ("electrode_group_name", "U32"),
        ("probe_shank", "i8"),
        ("original_reference_electrode", "i8"),
    ]
    rows = []
    for group_name in group_names:
        for _ in range(n_per_group):
            rows.append((len(rows), group_name, 1, 0))
    return np.array(rows, dtype=dtype)


class _FakeElectrode:
    """Stand-in for the Electrode table that returns a fixed fetch() result.

    get_group_by_shank only ever calls ``Electrode() & restr & restr`` then
    ``.fetch()``, so faking those three steps exercises the real grouping and
    sorting logic without a database.
    """

    def __init__(self, table):
        self._table = table

    def __call__(self):
        return self

    def __and__(self, _restriction):
        return self

    def fetch(self):
        return self._table


@pytest.fixture
def group_order(monkeypatch):
    """Return a helper mapping electrode group names to their sort order."""
    from spyglass.spikesorting import utils as sg_utils

    def _run(group_names):
        monkeypatch.setattr(
            sg_utils,
            "Electrode",
            _FakeElectrode(_electrode_array(group_names)),
        )
        _, sge_keys = sg_utils.get_group_by_shank(nwb_file_name="test.nwb")

        by_sort_group = {}
        for key in sge_keys:
            by_sort_group.setdefault(
                key["sort_group_id"], key["electrode_group_name"]
            )
        return [by_sort_group[i] for i in sorted(by_sort_group)]

    return _run


def test_natural_sort_key_orders_numbers_numerically():
    """Digit runs compare as numbers, not as text."""
    from spyglass.spikesorting.utils import natural_sort_key

    names = ["10", "2", "1", "0"]
    assert sorted(names, key=natural_sort_key) == ["0", "1", "2", "10"]


def test_natural_sort_key_accepts_non_numeric_names():
    """Descriptive names sort naturally instead of raising ValueError."""
    from spyglass.spikesorting.utils import natural_sort_key

    names = ["probe1_shank10", "probe1_shank2", "probe1_shank1"]
    assert sorted(names, key=natural_sort_key) == [
        "probe1_shank1",
        "probe1_shank2",
        "probe1_shank10",
    ]


def test_natural_sort_key_mixes_numeric_and_text_names():
    """A mix of numeric and descriptive names still yields a total order."""
    from spyglass.spikesorting.utils import natural_sort_key

    names = ["probe1_shank1", "2", "0"]
    assert sorted(names, key=natural_sort_key) == [
        "0",
        "2",
        "probe1_shank1",
    ]


def test_group_by_shank_numeric_groups_unchanged(group_order):
    """Numeric group names keep the numeric ordering they had before #1623."""
    assert group_order(["10", "2", "1", "0"]) == ["0", "1", "2", "10"]


def test_group_by_shank_non_numeric_groups(group_order):
    """#1623: a non-numeric electrode_group_name no longer raises ValueError."""
    assert group_order(
        ["probe1_shank10", "probe1_shank2", "probe1_shank1"]
    ) == ["probe1_shank1", "probe1_shank2", "probe1_shank10"]


def test_group_by_shank_mixed_groups(group_order):
    """#1623: numeric and descriptive names can coexist in one file."""
    assert group_order(["probe1_shank1", "2", "0"]) == [
        "0",
        "2",
        "probe1_shank1",
    ]
