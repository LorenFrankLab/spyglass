"""Regression coverage for ``fetch_spike_data`` on curated v1 units tables.

Covers zero-unit sortings (a bare Units table), sparse unit ids from
merge-applied curations, and unlabeled curations without a label column.
"""

from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


class _Relation:
    def __and__(self, restriction):
        return self


class _UnitsRelation(_Relation):
    def __init__(self, merge_ids):
        self.merge_ids = merge_ids

    def fetch(self, attribute):
        assert attribute == "spikesorting_merge_id"
        return self.merge_ids


class _ParamsRelation(_Relation):
    def __init__(self, include_labels=(), exclude_labels=()):
        self.include_labels = list(include_labels)
        self.exclude_labels = list(exclude_labels)

    def fetch1(self, *attributes):
        # ``fetch_spike_data`` fetches the whole row and reads the columns by
        # name, so ``unit_criteria`` can be absent on an un-altered table.
        assert attributes == ()
        return {
            "include_labels": self.include_labels,
            "exclude_labels": self.exclude_labels,
            "unit_criteria": None,
        }


class _MergeRelation(_Relation):
    def __init__(self, nwb_files, merge_ids):
        self.nwb_files = nwb_files
        self.merge_ids = merge_ids

    def fetch_nwb(self, *, return_merge_ids):
        assert return_merge_ids is True
        return self.nwb_files, self.merge_ids


def _fetch(monkeypatch, nwb_files, params=None):
    """Run the production ``fetch_spike_data`` over synthetic NWB payloads."""
    import spyglass.spikesorting.analysis.v1.group as module

    original = module.SortedSpikesGroup
    merge_ids = [uuid4() for _ in nwb_files]

    class _FakeSortedSpikesGroup:
        Units = _UnitsRelation(merge_ids)
        fetch_spike_data = classmethod(original.fetch_spike_data.__func__)

        @classmethod
        def get_fully_defined_key(cls, key):
            return key

        filter_units = staticmethod(original.filter_units)
        filter_units_by_criteria = staticmethod(
            original.filter_units_by_criteria
        )

    monkeypatch.setattr(module, "SortedSpikesGroup", _FakeSortedSpikesGroup)
    monkeypatch.setattr(
        module, "UnitSelectionParams", params or _ParamsRelation()
    )
    monkeypatch.setattr(
        module, "SpikeSortingOutput", _MergeRelation(nwb_files, merge_ids)
    )
    key = {"nwb_file_name": "units.nwb", "sorted_spikes_group_name": "g"}

    spike_times, unit_ids = _FakeSortedSpikesGroup.fetch_spike_data(
        key, return_unit_ids=True
    )
    return spike_times, unit_ids, merge_ids


def test_fetch_spike_data_skips_zero_unit_file(monkeypatch):
    """An empty Units table contributes nothing and later files still load."""
    expected_spikes = np.array([0.1, 0.2])
    nwb_files = [
        {"object_id": pd.DataFrame()},
        {
            "object_id": pd.DataFrame(
                {"spike_times": [expected_spikes]}, index=[17]
            )
        },
    ]

    spike_times, unit_ids, merge_ids = _fetch(monkeypatch, nwb_files)

    assert len(spike_times) == 1
    np.testing.assert_array_equal(spike_times[0], expected_spikes)
    assert unit_ids == [{"spikesorting_merge_id": merge_ids[1], "unit_id": 17}]


def test_fetch_spike_data_returns_nwb_unit_ids(monkeypatch):
    """Sparse NWB unit ids are returned as-is, not as positions."""
    nwb_files = [
        {
            "object_id": pd.DataFrame(
                {"spike_times": [np.array([0.2]), np.array([0.5])]},
                index=[2, 5],
            )
        }
    ]

    spike_times, unit_ids, merge_ids = _fetch(monkeypatch, nwb_files)

    assert [ids["unit_id"] for ids in unit_ids] == [2, 5]
    np.testing.assert_array_equal(spike_times[1], [0.5])


@pytest.mark.parametrize(
    "include_labels, exclude_labels, expected_ids",
    [
        (["accept"], [], []),
        ([], ["noise"], [3, 8]),
        ([], [], [3, 8]),
    ],
    ids=["include_only", "exclude_only", "no_labels"],
)
def test_fetch_spike_data_unlabeled_sorting(
    monkeypatch, include_labels, exclude_labels, expected_ids
):
    """Without a label column every unit carries no labels."""
    nwb_files = [
        {
            "object_id": pd.DataFrame(
                {"spike_times": [np.array([0.1]), np.array([0.2])]},
                index=[3, 8],
            )
        }
    ]
    params = _ParamsRelation(include_labels, exclude_labels)

    spike_times, unit_ids, _ = _fetch(monkeypatch, nwb_files, params)

    assert [ids["unit_id"] for ids in unit_ids] == expected_ids
    assert len(spike_times) == len(expected_ids)


@pytest.mark.parametrize(
    "include_labels, exclude_labels, n_warnings",
    [(["accept"], [], 1), ([], ["noise"], 0)],
    ids=["include_only", "exclude_only"],
)
def test_fetch_spike_data_unlabeled_sorting_warns(
    monkeypatch, caplog, include_labels, exclude_labels, n_warnings
):
    """include_labels on an unlabeled curation warns once per merge id."""
    nwb_files = [
        {
            "object_id": pd.DataFrame(
                {"spike_times": [np.array([0.1]), np.array([0.2])]},
                index=[3, 8],
            )
        }
    ]
    params = _ParamsRelation(include_labels, exclude_labels)

    _, _, merge_ids = _fetch(monkeypatch, nwb_files, params)

    warnings = [
        record.getMessage()
        for record in caplog.records
        if "include_labels removed all units" in record.getMessage()
    ]
    assert len(warnings) == n_warnings
    assert all(str(merge_ids[0]) in message for message in warnings)
