import numpy as np
import pytest

# Minimal subset of Electrode columns used by get_group_by_shank.
ELECTRODE_DTYPE = [
    ("electrode_group_name", "U32"),
    ("probe_shank", "i8"),
    ("original_reference_electrode", "i8"),
    ("electrode_id", "i8"),
]


class _FakeElectrode:
    """Stand-in for the Electrode table that returns canned rows from fetch()."""

    def __init__(self, records):
        self._records = records

    def __call__(self):
        return self

    def __and__(self, restriction):
        return self

    def fetch(self):
        return self._records


class TestGetGroupByShank:
    @pytest.fixture
    def run_group_by_shank(self, monkeypatch):
        """Run get_group_by_shank against canned Electrode rows.

        The Electrode DataJoint table is an external dependency, so it is
        patched with a fake that returns a structured array of electrode rows.
        """
        from spyglass.spikesorting import utils as spikesorting_utils

        def _run(rows, **kwargs):
            records = np.array(rows, dtype=ELECTRODE_DTYPE)
            monkeypatch.setattr(
                spikesorting_utils, "Electrode", _FakeElectrode(records)
            )
            return spikesorting_utils.get_group_by_shank("test.nwb", **kwargs)

        return _run

    def test_nonnumeric_group_names(self, run_group_by_shank):
        """Non-numeric electrode_group_name must not raise (regression for #1623).

        Previously get_group_by_shank sorted with `e_groups.sort(key=int)`, which
        raised ValueError on names like "probe1_shank1".
        """
        rows = [
            ("probe1_shank1", 1, -1, electrode_id) for electrode_id in range(4)
        ]
        sort_group_keys, sort_group_electrode_keys = run_group_by_shank(rows)

        assert sort_group_keys == [
            {
                "nwb_file_name": "test.nwb",
                "sort_group_id": 0,
                "sort_reference_electrode_id": -1,
            }
        ]
        assert [key["electrode_id"] for key in sort_group_electrode_keys] == [
            0,
            1,
            2,
            3,
        ]
        assert all(
            key["electrode_group_name"] == "probe1_shank1"
            for key in sort_group_electrode_keys
        )

    def test_numeric_group_names_sorted_numerically(self, run_group_by_shank):
        """Numeric names keep numeric ordering: 1, 2, 10 (not lexicographic 1, 10, 2)."""
        rows, electrode_id = [], 0
        for group_name in ["1", "2", "10"]:
            for _ in range(
                2
            ):  # two electrodes so the group is not omitted as a unitrode
                rows.append((group_name, 1, -1, electrode_id))
                electrode_id += 1
        _, sort_group_electrode_keys = run_group_by_shank(rows)

        group_name_by_sort_id = {}
        for key in sort_group_electrode_keys:
            group_name_by_sort_id.setdefault(
                key["sort_group_id"], key["electrode_group_name"]
            )
        assert group_name_by_sort_id == {0: "1", 1: "2", 2: "10"}
