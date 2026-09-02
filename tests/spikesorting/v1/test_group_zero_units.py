"""Regression coverage for zero-unit sorted-spikes groups."""

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
    def fetch1(self, *attributes):
        assert attributes == ("include_labels", "exclude_labels")
        return [], []


class _MergeRelation(_Relation):
    def __init__(self, nwb_files, merge_ids):
        self.nwb_files = nwb_files
        self.merge_ids = merge_ids

    def fetch_nwb(self, *, return_merge_ids, multi_source):
        assert return_merge_ids is True
        assert multi_source is True
        return self.nwb_files, self.merge_ids


def test_fetch_spike_data_skips_zero_unit_file(monkeypatch):
    """An empty Units table contributes nothing and later files still load."""
    import spyglass.spikesorting.analysis.v1.group as module

    original = module.SortedSpikesGroup
    empty_merge_id = uuid4()
    populated_merge_id = uuid4()
    expected_spikes = np.array([0.1, 0.2])
    nwb_files = [
        {"object_id": pd.DataFrame()},
        {
            "object_id": pd.DataFrame(
                {"spike_times": [expected_spikes]}, index=[17]
            )
        },
    ]

    class _FakeSortedSpikesGroup:
        Units = _UnitsRelation([empty_merge_id, populated_merge_id])
        fetch_spike_data = classmethod(original.fetch_spike_data.__func__)

        @classmethod
        def get_fully_defined_key(cls, key):
            return key

        filter_units = staticmethod(original.filter_units)

    monkeypatch.setattr(module, "SortedSpikesGroup", _FakeSortedSpikesGroup)
    monkeypatch.setattr(module, "UnitSelectionParams", _ParamsRelation())
    monkeypatch.setattr(
        module,
        "SpikeSortingOutput",
        _MergeRelation(nwb_files, [empty_merge_id, populated_merge_id]),
    )
    key = {
        "nwb_file_name": "zero-units.nwb",
        "sorted_spikes_group_name": "mixed",
    }

    spike_times, unit_ids = _FakeSortedSpikesGroup.fetch_spike_data(
        key, return_unit_ids=True
    )

    assert len(spike_times) == 1
    np.testing.assert_array_equal(spike_times[0], expected_spikes)
    assert unit_ids == [
        {"spikesorting_merge_id": populated_merge_id, "unit_id": 17}
    ]
