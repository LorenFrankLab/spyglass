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


@pytest.fixture
def plant_members(merge_table, monkeypatch):
    """Plant the trains, pipeline source, and observed spans of each merge.

    A member is ``{"merge_id", "source", "intervals", "trains"}`` where
    ``trains`` maps unit id to spike times and ``intervals`` is the span list
    the merge's curated NWB would report (``None`` for a source that stores
    none). Returns the list of ``selection_observations`` calls the code makes,
    so a test can assert no NWB read is attempted for a legacy source.
    """
    from spyglass.spikesorting.v2 import _observation_io

    def _plant(members):
        spike_times, unit_ids, observed = [], [], {}
        for member in members:
            observed[member["merge_id"]] = member["intervals"]
            for unit_id, times in member["trains"].items():
                spike_times.append(np.asarray(times, dtype=float))
                unit_ids.append(
                    {
                        "spikesorting_merge_id": member["merge_id"],
                        "unit_id": unit_id,
                    }
                )
        monkeypatch.setattr(
            merge_table,
            "get_spike_times_by_unit",
            lambda self, key: (spike_times, unit_ids),
        )
        monkeypatch.setattr(
            merge_table,
            "_merge_source_map",
            staticmethod(
                lambda merge_ids: {
                    str(member["merge_id"]): member["source"]
                    for member in members
                }
            ),
        )

        calls = []

        def fake_selection_observations(merge_id, requested):
            calls.append((merge_id, sorted(requested)))
            intervals = observed[merge_id]
            if intervals is None:
                return None
            return {
                "masks": [
                    np.asarray(intervals, dtype=float).reshape(-1, 2).tolist()
                ],
                "unit_mask_ids": {str(unit_id): 0 for unit_id in requested},
            }

        monkeypatch.setattr(
            _observation_io,
            "selection_observations",
            fake_selection_observations,
        )
        return calls

    return _plant


def _v2(merge_id, intervals, trains):
    return {
        "merge_id": merge_id,
        "source": "CurationV2",
        "intervals": intervals,
        "trains": trains,
    }


def _v1(merge_id, trains):
    return {
        "merge_id": merge_id,
        "source": "CurationV1",
        "intervals": None,
        "trains": trains,
    }


def test_merge_indicator_marks_unobserved_bins(merge_table, plant_members):
    """A v2 merge's unobserved bin is NaN; a legacy merge's stays zero."""
    time = np.arange(10.0)
    trains = {1: [1.0, 3.5, 5.0]}

    calls = plant_members([_v2("v2-a", [[0, 3], [4, 10]], trains)])
    indicator, valid = merge_table.get_spike_indicator(
        {}, time, return_validity=True
    )

    assert calls == [("v2-a", [1])]
    assert np.isnan(indicator[:, 0]).tolist() == [
        i == 3 for i in range(len(time))
    ]
    assert valid.tolist() == [i != 3 for i in range(len(time))]
    # The 3.5 spike falls in the unobserved gap and is not counted anywhere.
    assert np.flatnonzero(indicator[:, 0] == 1).tolist() == [1, 5]
    assert np.nansum(indicator) == 2

    calls = plant_members([_v1("v1-a", trains)])
    indicator = merge_table.get_spike_indicator({}, time)
    observation = merge_table.get_observation_intervals({})

    assert calls == []  # a legacy source is never read for spans
    assert np.isfinite(indicator).all()
    assert np.flatnonzero(indicator[:, 0] == 1).tolist() == [1, 3, 5]
    assert observation.intervals is None
    assert observation.unknown_sources == ("v1-a",)


def test_merge_indicator_two_v2_members_with_different_gaps(
    merge_table, plant_members
):
    """Every unit is filtered by the common span, not by its own member's."""
    time = np.arange(10.0)
    plant_members(
        [
            _v2("v2-a", [[0, 3], [4, 10]], {1: [1.0, 6.5]}),
            _v2("v2-b", [[0, 6], [7, 10]], {2: [8.0]}),
        ]
    )

    indicator, unit_ids, valid = merge_table.get_spike_indicator(
        {}, time, return_unit_ids=True, return_validity=True
    )

    assert unit_ids == [
        {"spikesorting_merge_id": "v2-a", "unit_id": 1},
        {"spikesorting_merge_id": "v2-b", "unit_id": 2},
    ]
    assert indicator.shape == (10, 2)
    # Each member's gap invalidates a bin for the whole population.
    assert np.flatnonzero(~valid).tolist() == [3, 6]
    assert np.isnan(indicator[[3, 6], :]).all()
    # v2-a's 6.5 spike sits inside v2-a's own coverage but outside the common
    # span, so it is dropped rather than moved into an adjacent valid bin.
    assert np.nansum(indicator[:, 0]) == 1
    assert indicator[1, 0] == 1
    assert indicator[5, 0] == 0
    assert indicator[7, 0] == 0
    assert np.flatnonzero(indicator[:, 1] == 1).tolist() == [8]


def test_merge_indicator_endpoint_spike_outside_common_availability(
    merge_table, plant_members
):
    """The final bin's zero-width end cannot admit a spike outside the span.

    ``np.digitize(times, time[1:-1])`` puts a spike at the axis's last
    timestamp in the preceding bin, which is valid here. Filtering by the
    member's own coverage would therefore count v2-a's 3.0 spike in bin 2;
    filtering by the common span drops it.
    """
    time = np.array([0.0, 1.0, 2.0, 3.0])
    plant_members(
        [
            _v2("v2-a", [[0, 4]], {1: [3.0]}),
            _v2("v2-b", [[0, 3]], {2: [1.0]}),
        ]
    )

    indicator = merge_table.get_spike_indicator({}, time)

    assert np.isnan(indicator[3, :]).all()
    assert indicator[:3, 0].tolist() == [0, 0, 0]
    assert indicator[:3, 1].tolist() == [0, 1, 0]


def test_merge_indicator_mixed_legacy_and_v2(merge_table, plant_members):
    """A known member restricts the whole population; the unknown one is named."""
    time = np.arange(10.0)
    plant_members(
        [
            _v1("v1-a", {1: [5.0]}),
            _v2("v2-b", [[0, 3], [4, 10]], {2: [1.0]}),
        ]
    )

    indicator = merge_table.get_spike_indicator({}, time)
    observation = merge_table.get_observation_intervals({})

    assert np.flatnonzero(np.isnan(indicator[:, 0])).tolist() == [3]
    assert np.flatnonzero(np.isnan(indicator[:, 1])).tolist() == [3]
    assert indicator[5, 0] == 1
    assert indicator[1, 1] == 1
    np.testing.assert_array_equal(observation.intervals, [[0, 3], [4, 10]])
    assert observation.unknown_sources == ("v1-a",)


def test_merge_indicator_no_matching_member(merge_table, plant_members):
    """A restriction matching nothing returns an empty, fully valid result."""
    time = np.arange(10.0)
    plant_members([])

    indicator, valid = merge_table.get_spike_indicator(
        {}, time, return_validity=True
    )

    assert indicator.shape == (10, 0)
    assert valid.all()
    assert merge_table.get_firing_rate({}, time).shape == (10, 0)


def test_merge_firing_rate_splits_a_jump_with_every_bin_observed(
    merge_table, plant_members
):
    """A fully observed merge still splits smoothing at a timestamp jump.

    The observed span covers the whole axis, so every bin is valid and the
    validity mask gives no reason to split; only the six-second clock jump
    does. Smoothing the whole axis would bleed the single spike across it.
    """
    time = np.r_[np.arange(4.0), 10.0 + np.arange(4.0)]
    plant_members([_v2("v2-a", [[0, 14]], {1: [1.0]})])

    indicator, valid = merge_table.get_spike_indicator(
        {}, time, return_validity=True
    )
    assert valid.all()
    assert np.flatnonzero(indicator[:, 0] == 1).tolist() == [1]

    rate = merge_table.get_firing_rate({}, time, smoothing_sigma=2)

    assert np.isfinite(rate).all()
    assert rate[:4, 0].sum() > 0
    assert rate[4:, 0].tolist() == [0.0] * 4


def test_merge_observation_failure_names_the_merge(
    merge_table, plant_members, monkeypatch
):
    """An unreadable recording raises a message pointing somewhere useful.

    Reporting full coverage instead would hide exactly the defect the
    observed-time contract exists to catch, so the read still raises -- but
    it names the merge and the snapshot-backed alternative.
    """
    from spyglass.spikesorting.v2 import _observation_io

    plant_members([_v2("v2-a", [[0, 3], [4, 10]], {1: [1.0]})])

    def unreadable(merge_id, requested):
        raise FileNotFoundError("recording folder is gone")

    monkeypatch.setattr(_observation_io, "selection_observations", unreadable)

    with pytest.raises(RuntimeError, match="v2-a") as raised:
        merge_table.get_spike_indicator({}, np.arange(10.0))

    assert isinstance(raised.value.__cause__, FileNotFoundError)
    assert "SortedSpikesGroup" in str(raised.value)


def test_merge_firing_rate_does_not_bleed_into_gap(merge_table, plant_members):
    """Smoothing runs per observed span, so no rate crosses the gap."""
    time = np.arange(10.0)
    plant_members([_v2("v2-a", [[0, 3], [4, 10]], {1: [1.0]})])

    rate = merge_table.get_firing_rate({}, time, smoothing_sigma=2)

    assert rate.shape == (10, 1)
    assert np.isnan(rate[3, 0])
    assert np.isfinite(rate[[0, 1, 2, 4, 5, 6, 7, 8, 9], 0]).all()
    assert rate[:3, 0].sum() > 0
    # The only spike is on the far side of the gap: the bins after it stay at
    # zero instead of inheriting smoothed density through unobserved time.
    assert rate[4:, 0].tolist() == [0.0] * 6


def test_merge_consumers_finite_on_observed_slices(
    merge_table, plant_members, monkeypatch
):
    """A NaN-bearing rate reaches the MUA figure as finite plotted samples.

    ``create_figurl`` z-scores and plots only the observed samples, so the
    NaN bins the merge accessor reports never reach ``zscore`` or the view,
    while the accessor's own output keeps them.
    """
    import sortingview.views as real_views

    from spyglass.mua.v1 import mua as mua_module

    time = np.arange(10.0)
    plant_members([_v2("v2-a", [[0, 3], [4, 10]], {1: [1.0, 5.0, 5.5]})])
    rate = merge_table.get_firing_rate(
        {}, time, multiunit=True, smoothing_sigma=2
    )
    assert np.isnan(rate[3, 0])

    zscored_inputs = []
    real_zscore = mua_module.zscore

    def recording_zscore(values, **kwargs):
        zscored_inputs.append(np.asarray(values, dtype=float))
        return real_zscore(values, **kwargs)

    graphs = []

    class _RecordingGraph(real_views.TimeseriesGraph):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            graphs.append(self)

    class _FakeBox:
        def url(self, label=None):
            return f"fake://{label}"

    class _FakeViews:
        TimeseriesGraph = _RecordingGraph
        TGDataset = real_views.TGDataset
        TGSeries = real_views.TGSeries

        @staticmethod
        def LayoutItem(*args, **kwargs):
            return object()

        @staticmethod
        def Box(**kwargs):
            return _FakeBox()

    class _FakeParams:
        def __and__(self, restriction):
            return self

        def fetch1(self, name):
            return {"zscore_threshold": 2.0}

    monkeypatch.setattr(mua_module, "zscore", recording_zscore)
    monkeypatch.setattr(mua_module, "vv", _FakeViews)
    monkeypatch.setattr(mua_module, "MuaEventsParameters", _FakeParams())
    monkeypatch.setattr(
        mua_module.MuaEventsV1, "fetch1", staticmethod(lambda *a: {})
    )
    monkeypatch.setattr(
        mua_module.MuaEventsV1,
        "get_speed",
        staticmethod(lambda key: pd.Series(np.zeros(time.size), index=time)),
    )
    monkeypatch.setattr(
        mua_module.MuaEventsV1,
        "get_firing_rate",
        staticmethod(lambda key, t: rate),
    )
    monkeypatch.setattr(
        mua_module.MuaEventsV1,
        "fetch1_dataframe",
        staticmethod(
            lambda: pd.DataFrame({"start_time": [5.0], "end_time": [6.0]})
        ),
    )

    url = mua_module.MuaEventsV1().create_figurl()

    assert url == "fake://Multiunit Detection"
    assert len(zscored_inputs) == 1
    assert zscored_inputs[0].size == 9  # every bin but the unobserved one
    assert np.isfinite(zscored_inputs[0]).all()
    graph = graphs[0].to_dict()
    datasets = {data["name"]: data["data"] for data in graph["datasets"]}
    rate_series = [
        series
        for series in graph["series"]
        if series["dataset"].startswith("Z-Scored Multiunit Rate")
    ]
    assert len(rate_series) == 2
    drawn_times, drawn_rates = [], []
    for series in rate_series:
        data = datasets[series["dataset"]]
        times = data["t"].astype(float) + graph["timeOffset"]
        assert np.all(np.diff(times) == 1)
        assert np.isfinite(data["y"]).all()
        drawn_times.append(times)
        drawn_rates.append(data["y"])
    valid = np.isfinite(rate[:, 0])
    np.testing.assert_array_equal(np.concatenate(drawn_times), time[valid])
    np.testing.assert_allclose(
        np.concatenate(drawn_rates), real_zscore(rate[valid, 0]), rtol=1e-6
    )
    # The public accessor still reports the unobserved bin.
    assert np.isnan(rate[3, 0])
