"""Merge-level spike accessors keep unit identity and observed-time limits.

``SpikeSortingOutput`` aggregates spike trains across merge entries, so a bin
is only evidence of "no spikes" when every contributing unit was observed over
it. These tests plant trains and observation masks directly: the merge table's
schema has to be declared (hence the ``common`` fixture's connection), but no
row, NWB file, or recording is read.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def merge_table(common):
    """The merge table class, imported once its schema can be declared."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    return SpikeSortingOutput


@pytest.fixture
def silence_merge_warnings(merge_table, monkeypatch):
    """Record the merge ids each warning helper saw instead of querying."""
    seen = {}

    def recorder(name):
        # staticmethod: one helper is called on the class and the other on the
        # instance, so neither may receive an implicit first argument.
        return staticmethod(
            lambda merge_ids: seen.setdefault(name, list(merge_ids))
        )

    for name in ("_warn_preview_merge_ids", "_warn_multi_source_merge_ids"):
        monkeypatch.setattr(merge_table, name, recorder(name))
    return seen


def _nwb_file(field, trains_by_unit):
    """A fetch_nwb-shaped mapping whose units frame is indexed by unit id."""
    return {
        field: pd.DataFrame(
            {"spike_times": [np.asarray(t) for t in trains_by_unit.values()]},
            index=list(trains_by_unit),
        )
    }


def test_merge_spike_times_by_unit_keeps_identities(
    merge_table, monkeypatch, silence_merge_warnings
):
    """Each train carries the merge it came from and its true unit id.

    The second file uses sparse ids (a v2 merge-applied sorting drops the
    contributor ids), and the third is a zero-unit curation that contributes
    no trains and no identities.
    """
    files = [
        _nwb_file("object_id", {1: [0.5], 3: [1.5, 2.5]}),
        _nwb_file("units", {7: [3.5]}),
        {"object_id": pd.DataFrame(index=[])},
    ]
    monkeypatch.setattr(
        merge_table,
        "fetch_nwb",
        lambda self, key, **kwargs: (files, ["merge-a", "merge-b", "merge-c"]),
    )

    spike_times, unit_ids = merge_table().get_spike_times_by_unit({})

    assert [times.tolist() for times in spike_times] == [
        [0.5],
        [1.5, 2.5],
        [3.5],
    ]
    assert unit_ids == [
        {"spikesorting_merge_id": "merge-a", "unit_id": 1},
        {"spikesorting_merge_id": "merge-a", "unit_id": 3},
        {"spikesorting_merge_id": "merge-b", "unit_id": 7},
    ]
    assert silence_merge_warnings == {
        "_warn_preview_merge_ids": ["merge-a", "merge-b", "merge-c"],
        "_warn_multi_source_merge_ids": ["merge-a", "merge-b", "merge-c"],
    }


def test_merge_get_spike_times_returns_bare_trains(
    merge_table, monkeypatch, silence_merge_warnings
):
    """The public accessor still returns just the list of trains."""
    monkeypatch.setattr(
        merge_table,
        "fetch_nwb",
        lambda self, key, **kwargs: (
            [_nwb_file("object_id", {2: [0.25]})],
            ["merge-a"],
        ),
    )

    spike_times = merge_table().get_spike_times({})

    assert isinstance(spike_times, list)
    assert [times.tolist() for times in spike_times] == [[0.25]]
