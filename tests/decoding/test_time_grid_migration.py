"""Hermetic actual-adapter tests; run with --noconftest to avoid DB fixtures.

These execute the real source methods and real decoder, not mock decoder calls.
The DataJoint table module is not imported. This is API and row-alignment
coverage, not a replacement for the separate database/NWB integration suite.
"""

import ast
import importlib.util
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import non_local_detector.analysis as analysis
from non_local_detector import (
    Environment,
    ContFragSortedSpikesClassifier,
    ContFragClusterlessClassifier,
)
from non_local_detector.models.base import (
    SortedSpikesDetector,
    ClusterlessDetector,
)
from non_local_detector.continuous_state_transitions import Uniform
from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryDiagonal,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/spyglass/decoding/v1"
spec = importlib.util.spec_from_file_location(
    "isolated_spyglass_utils", SOURCE / "utils.py"
)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def source_method(filename, class_name, method_name):
    tree = ast.parse((SOURCE / filename).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    node = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    namespace = vars(utils).copy() | {
        "SortedSpikesDetector": SortedSpikesDetector,
        "ClusterlessDetector": ClusterlessDetector,
        "logger": logging.getLogger("time-grid-test"),
        "analysis": analysis,
        "pd": pd,
    }
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(SOURCE / filename),
            "exec",
        ),
        namespace,
    )
    return namespace[method_name]


def fixture(family, tracking_rate=30, covariates=False):
    time = np.arange(tracking_rate + 1) / tracking_rate
    tracking = pd.DataFrame({"position": 5 + np.sin(time * 6)}, index=time)
    tracking.attrs["valid_position_intervals"] = [[0.0, 1.0]]
    spikes = [np.array([0.019, 0.101, 0.201, 0.501, 0.801])]
    marks = [
        np.array([[2.0, 4.0], [5.0, 3.0], [7.0, 2.0], [4.0, 6.0], [3.0, 5.0]])
    ]
    cls, base = (
        (ContFragSortedSpikesClassifier, SortedSpikesDetector)
        if family == "sorted"
        else (ContFragClusterlessClassifier, ClusterlessDetector)
    )
    args = dict(
        environments=Environment(place_bin_size=2, position_range=((0, 10),)),
        infer_track_interior=False,
        continuous_transition_types=[
            [Uniform(), Uniform()],
            [Uniform(), Uniform()],
        ],
    )
    if covariates:
        args["discrete_transition_type"] = DiscreteNonStationaryDiagonal(
            np.array([0.9, 0.9]), formula="1 + speed"
        )
    template = cls(**args)
    params = {
        name: getattr(template, name)
        for name in inspect.signature(base).parameters
    }
    return tracking, spikes, marks, params


def run_adapter(
    family,
    estimate,
    tracking,
    spikes,
    marks,
    params,
    kwargs=None,
    intervals=None,
):
    filename = "sorted_spikes.py" if family == "sorted" else "clusterless.py"
    name = (
        "SortedSpikesDecodingV1"
        if family == "sorted"
        else "ClusterlessDecodingV1"
    )
    method = source_method(filename, name, "_run_decoder")
    args = [
        None,
        {"estimate_decoding_params": estimate},
        params,
        kwargs or {},
        tracking,
        ["position"],
        spikes,
    ]
    if family == "clusterless":
        args.append(marks)
    args.append(
        np.array([[0.1, 0.3]]) if intervals is None else np.asarray(intervals)
    )
    return method(*args)


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
@pytest.mark.parametrize("estimate", [False, True])
@pytest.mark.parametrize("tracking_rate", [30, 500])
def test_actual_adapters_use_uniform_grid_and_actual_interval_rows(
    family, estimate, tracking_rate
):
    tracking, spikes, marks, params = fixture(family, tracking_rate)
    kwargs = {"max_iter": 1} if estimate else {}
    model, results = run_adapter(
        family, estimate, tracking, spikes, marks, params, kwargs
    )
    assert results.sizes["time"] == (500 if estimate else 100)
    if not estimate:
        # Five supported events over one physical second, independent of camera
        # rate. This catches accidental restoration of per-sample rates.
        np.testing.assert_allclose(
            model.encoding_model_[("", 0)]["mean_rates"], [5.0], rtol=1e-6
        )
    np.testing.assert_allclose(
        results.time_bin_end - results.time_bin_start, 0.002, rtol=1e-12
    )
    assert results.interval_labels.shape == results.time.shape
    assert np.all(
        results.interval_labels.values[~results.is_missing.values] == 0
    )
    np.testing.assert_allclose(
        results.acausal_posterior.sum("state_bins"), 1.0, rtol=1e-6
    )


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_nan_tracking_does_not_change_interval_identity(family):
    tracking, spikes, marks, params = fixture(family)
    tracking.iloc[15, 0] = np.nan
    _, results = run_adapter(
        family, True, tracking, spikes, marks, params, {"max_iter": 1}, [[0, 1]]
    )
    assert results.is_missing.any()
    assert np.all(results.interval_labels.values == 0)


@pytest.mark.parametrize(
    "fetch_order, second_times, overlaps",
    [
        ([0, 1], [10.0, 11.0], False),
        ([1, 0], [10.0, 11.0], False),
        ([1, 0], [0.5, 1.5], True),
    ],
)
def test_epoch_support_is_preserved_by_actual_position_fetch(
    fetch_order, second_times, overlaps
):
    frames = {
        0: pd.DataFrame({"position": [0.0, 1.0]}, index=[0.0, 1.0]),
        1: pd.DataFrame({"position": [2.0, 3.0]}, index=second_times),
    }

    class PositionRelation:
        def __and__(self, key):
            return SimpleNamespace(
                fetch1_dataframe=lambda: frames[key["merge_id"]]
            )

    class Group:
        Position = SimpleNamespace(__and__=lambda self, key: self)

        def __and__(self, key):
            return self

        def fetch1(self, name):
            return {
                "KEY": {},
                "position_variables": ["position"],
                "upsample_rate": np.nan,
            }[name]

    class Part:
        def __and__(self, key):
            return self

        def fetch(self, name):
            return fetch_order

    group = Group()
    group.Position = Part()
    method = source_method("core.py", "PositionGroup", "fetch_position_info")
    method.__globals__["PositionOutput"] = PositionRelation()
    data, names = method(group)
    assert names == ["position"]
    expected = [[0.0, 1.0], second_times]
    assert data.attrs["valid_position_intervals"] == expected
    assert data.index.tolist() == sorted([0.0, 1.0, *second_times])
    if overlaps:
        with pytest.raises(ValueError, match="ordered and non-overlapping"):
            utils.declared_tracking_intervals(data)
    else:
        np.testing.assert_array_equal(
            utils.declared_tracking_intervals(data), expected
        )


def test_adjacent_sequences_count_shared_event_once_and_keep_marks():
    spikes = [np.array([0.1, 0.2, 0.3])]
    marks = [np.array([[1.0], [2.0], [3.0]])]
    first, first_marks = utils.spikes_for_sequence(
        spikes,
        np.array([0.1, 0.2]),
        shared_stop=True,
        spike_waveform_features=marks,
    )
    second, second_marks = utils.spikes_for_sequence(
        spikes, np.array([0.2, 0.3]), spike_waveform_features=marks
    )
    np.testing.assert_array_equal(np.concatenate(first + second), spikes[0])
    np.testing.assert_array_equal(
        np.concatenate(first_marks + second_marks), marks[0]
    )


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_timestamped_covariates_are_aligned_for_actual_prediction(family):
    tracking, spikes, marks, params = fixture(family, covariates=True)
    covariates = pd.DataFrame(
        {"speed": np.linspace(0, 20, len(tracking))}, index=tracking.index
    )
    _, results = run_adapter(
        family,
        False,
        tracking,
        spikes,
        marks,
        params,
        {"discrete_transition_covariate_data": covariates},
    )
    assert results.sizes["time"] == 100
    assert results.discrete_state_transitions.dims == (
        "transition_time",
        "states_from",
        "states_to",
    )
    np.testing.assert_array_equal(results.transition_time, tracking.index)


def test_grid_preparation_rejects_covariate_invention_in_masked_gaps():
    tracking, _, _, params = fixture("sorted")
    tracking.iloc[15, 0] = np.nan
    covariates = pd.DataFrame(
        {"speed": np.ones(len(tracking))}, index=tracking.index
    )
    with pytest.raises(ValueError, match="finite even in masked HMM gaps"):
        utils.prepare_decoder_grid(
            SortedSpikesDetector(**params),
            tracking,
            ["position"],
            [0, 1],
            {"discrete_transition_covariate_data": covariates},
        )
    centers = np.arange(500) * 0.002 + 0.001
    explicit = pd.DataFrame({"speed": np.ones(500)}, index=centers)
    # Use the exact centers returned by the grid rather than relying on bitwise
    # equality of another floating-point construction.
    classifier = SortedSpikesDetector(**params)
    edges = classifier.calculate_time_edges([0, 1])
    explicit.index = (edges[:-1] + edges[1:]) / 2
    _, missing, actual = utils.prepare_decoder_grid(
        classifier,
        tracking,
        ["position"],
        [0, 1],
        {"discrete_transition_covariate_data": explicit},
    )
    assert missing.any()
    assert not actual.isna().any().any()


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_actual_ahead_behind_consumer_aligns_31_samples_to_500_bins(family):
    tracking, spikes, marks, params = fixture(family)
    classifier, results = run_adapter(
        family, False, tracking, spikes, marks, params, intervals=[[0, 1]]
    )
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(100.0, 0.0))
    graph.add_edge(0, 1, distance=100.0)
    classifier.environments[0].track_graph = graph
    time = tracking.index.to_numpy()
    linear = pd.DataFrame(
        {
            "projected_x_position": 100 * time,
            "projected_y_position": 0.0,
            "track_segment_id": 0,
            "head_orientation": 0.0,
        },
        index=time,
    )
    linear.attrs.update(tracking.attrs)
    dummy = SimpleNamespace(
        fetch_model=lambda: classifier,
        fetch_results=lambda: results,
        fetch_linear_position_info=lambda key: linear,
        fetch1=lambda key: {},
        get_orientation_col=lambda data: "head_orientation",
    )
    filename = "sorted_spikes.py" if family == "sorted" else "clusterless.py"
    name = (
        "SortedSpikesDecodingV1"
        if family == "sorted"
        else "ClusterlessDecodingV1"
    )
    method = source_method(filename, name, "get_ahead_behind_distance")
    mental = np.column_stack([100 * results.time.values, np.zeros(500)])
    with patch.object(
        analysis.distance1D,
        "_get_MAP_estimate_2d_position_edges",
        return_value=(mental, np.tile([0, 1], (500, 1))),
    ):
        distance = method(dummy)
    assert distance.shape == (500,)
    np.testing.assert_allclose(distance, 0.0, atol=1e-12)


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
@pytest.mark.parametrize("estimate", [False, True])
def test_synthetic_nwb_arrays_feed_actual_adapters_without_a_database(
    tmp_path, family, estimate
):
    from datetime import datetime, timezone
    import datajoint  # Installed supported dependency; no connection/schema call.
    from pynwb import NWBFile, NWBHDF5IO, TimeSeries

    assert datajoint.__version__
    tracking, spikes, marks, params = fixture(family)
    nwb = NWBFile(
        session_description="Hermetic decoder migration fixture",
        identifier="time-grid-fixture",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    nwb.add_acquisition(
        TimeSeries(
            name="tracking",
            data=tracking[["position"]].to_numpy(),
            unit="cm",
            timestamps=tracking.index.to_numpy(),
        )
    )
    nwb.add_acquisition(
        TimeSeries(name="marks", data=marks[0], unit="mV", timestamps=spikes[0])
    )
    nwb.add_unit(spike_times=spikes[0])
    nwb.add_epoch(start_time=0.0, stop_time=1.0, tags=["continuous_tracking"])
    path = tmp_path / "migration.nwb"
    with NWBHDF5IO(path, "w") as io:
        io.write(nwb)
    with NWBHDF5IO(path, "r") as io:
        restored = io.read()
        position = restored.acquisition["tracking"]
        fetched = pd.DataFrame(
            position.data[:], index=position.timestamps[:], columns=["position"]
        )
        fetched.attrs[
            "valid_position_intervals"
        ] = restored.epochs.to_dataframe()[
            ["start_time", "stop_time"]
        ].to_numpy()
        fetched_spikes = [restored.units.get_unit_spike_times(0)]
        fetched_marks = [restored.acquisition["marks"].data[:]]
    model, results = run_adapter(
        family,
        estimate,
        fetched,
        fetched_spikes,
        fetched_marks,
        params,
        {"max_iter": 1} if estimate else {},
    )
    assert model.encoding_model_[("", 0)]["rate_units"] == "Hz"
    assert results.sizes["time"] == (500 if estimate else 100)
    np.testing.assert_allclose(
        results.time_bin_end - results.time_bin_start, 0.002, rtol=1e-12
    )


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_actual_adjacent_adapter_sequences_own_the_declared_boundary_once(
    family,
):
    from functools import wraps

    tracking, spikes, marks, params = fixture(family)
    spikes = [np.insert(spikes[0], 3, 0.3)]
    marks = [np.insert(marks[0], 3, [9.0, 10.0], axis=0)]
    base = SortedSpikesDetector if family == "sorted" else ClusterlessDetector
    original = base.predict
    passed = []

    @wraps(original)
    def record(self, *args, **kwargs):
        passed.append(
            (kwargs["spike_times"], kwargs.get("spike_waveform_features"))
        )
        return original(self, *args, **kwargs)

    with patch.object(base, "predict", record):
        _, results = run_adapter(
            family,
            False,
            tracking,
            spikes,
            marks,
            params,
            intervals=[[0.1, 0.3], [0.3, 0.5]],
        )
    assert results.sizes["time"] == 200
    all_passed = np.concatenate([unit[0] for unit, _ in passed])
    assert (all_passed == 0.3).sum() == 1
    assert 0.3 not in passed[0][0][0]
    assert 0.3 in passed[1][0][0]
    if family == "clusterless":
        owner = np.flatnonzero(passed[1][0][0] == 0.3)[0]
        np.testing.assert_array_equal(passed[1][1][0][owner], [9.0, 10.0])


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_between_camera_interval_decodes_all_fully_supported_bins(family):
    tracking, spikes, marks, params = fixture(family, tracking_rate=30)
    assert not ((tracking.index >= 0.004) & (tracking.index <= 0.010)).any()
    _, results = run_adapter(
        family,
        False,
        tracking,
        spikes,
        marks,
        params,
        intervals=[[0.004, 0.010]],
    )
    assert results.sizes["time"] == 3
    assert not results.is_missing.any()


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_sub_bin_and_unsupported_intervals_do_not_drop_later_valid_results(
    family,
):
    tracking, spikes, marks, params = fixture(family, tracking_rate=30)
    _, results = run_adapter(
        family,
        False,
        tracking,
        spikes,
        marks,
        params,
        intervals=[[-0.1, -0.01], [0.004, 0.005], [0.010, 0.020]],
    )
    assert results.sizes["time"] == 5
    assert np.all(results.interval_labels.values == 2)
    assert not results.is_missing.any()


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
@pytest.mark.parametrize("estimate", [False, True])
def test_original_tracking_masks_are_aligned_on_original_timeline(
    family, estimate
):
    tracking, spikes, marks, params = fixture(family, tracking_rate=30)
    missing = np.zeros(len(tracking), bool)
    missing[5] = True
    kwargs = {
        "is_missing": missing,
        "is_training": np.ones(len(tracking), bool),
    }
    if estimate:
        kwargs["max_iter"] = 1
    _, results = run_adapter(
        family, estimate, tracking, spikes, marks, params, kwargs, [[0, 1]]
    )
    assert results.sizes["time"] == 500
    expected = (results.time.values >= tracking.index[5]) & (
        results.time.values < tracking.index[6]
    )
    np.testing.assert_array_equal(results.is_missing, expected)
    # Missingness does not relabel the requested observed interval.
    assert np.all(results.interval_labels.values == 0)


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_adapter_preserves_scalar_missing_mask_support(family):
    tracking, spikes, marks, params = fixture(family)
    _, results = run_adapter(
        family, False, tracking, spikes, marks, params, {"is_missing": False}
    )
    assert results.sizes["time"] == 100
    assert not results.is_missing.any()


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_shifted_origin_interval_labels_match_prediction_and_em(family):
    tracking, spikes, marks, params = fixture(family)
    tracking.index = tracking.index + 0.1
    tracking.attrs["valid_position_intervals"] = [[0.1, 1.1]]
    spikes = [unit + 0.1 for unit in spikes]
    _, predicted = run_adapter(
        family, False, tracking, spikes, marks, params, intervals=[[0.1, 0.3]]
    )
    _, estimated = run_adapter(
        family,
        True,
        tracking,
        spikes,
        marks,
        params,
        {"max_iter": 1},
        [[0.1, 0.3]],
    )
    assert predicted.sizes["time"] == 100
    assert (estimated.interval_labels.values == 0).sum() == 100
    assert (~estimated.is_missing.values).sum() == 100


def test_interval_label_roundoff_never_admits_materially_partial_bins():
    from non_local_detector import calculate_time_edges

    edges = calculate_time_edges([0.1, 0.3], 500)
    assert (
        utils.decoder_interval_labels(edges, [[0.1, 0.3]]) >= 0
    ).sum() == 100
    assert (
        utils.decoder_interval_labels(edges, [[0.1, 0.299]]) >= 0
    ).sum() == 99


def test_physical_interval_labels_keep_host_workspace_independent_of_window_count():
    import tracemalloc
    from non_local_detector import calculate_time_edges

    edges = calculate_time_edges([0, 36], 500)

    def peak(n_windows):
        intervals = np.column_stack(
            [np.linspace(0, 34, n_windows), np.linspace(1, 35, n_windows)]
        )
        tracemalloc.start()
        utils.decoder_interval_labels(edges, intervals)
        _, maximum = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return maximum

    assert peak(50) < peak(10) * 2


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
@pytest.mark.parametrize("estimate", [False, True])
def test_serialized_concrete_model_instance_reaches_actual_adapter(
    monkeypatch, family, estimate
):
    """Exercise real parameter insert/fetch and decoding without table I/O."""
    path = Path(__file__).with_name("test_dj_decoder_conversion.py")
    spec = importlib.util.spec_from_file_location(
        "parameter_roundtrip_tests", path
    )
    parameter_tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parameter_tests)
    modules = parameter_tests._load_decoding_modules(monkeypatch)
    tracking, spikes, marks, params = fixture(family)
    cls = (
        ContFragSortedSpikesClassifier
        if family == "sorted"
        else ContFragClusterlessClassifier
    )
    original = cls(
        **{
            name: value
            for name, value in params.items()
            if name in inspect.signature(cls).parameters
        }
    )
    table = modules.core.DecodingParameters()
    table.insert(
        [
            {
                "decoding_param_name": "time-grid-roundtrip",
                "decoding_params": original,
                "decoding_kwargs": {},
            }
        ]
    )
    restored = table.fetch1("decoding_params")
    assert type(restored) is cls
    assert restored is not original
    model, results = run_adapter(
        family,
        estimate,
        tracking,
        spikes,
        marks,
        restored,
        {"max_iter": 1} if estimate else {},
    )
    assert model is restored
    assert type(model) is cls
    assert model.encoding_model_[("", 0)]["rate_units"] == "Hz"
    assert results.sizes["time"] == (500 if estimate else 100)
    np.testing.assert_allclose(
        results.time_bin_end - results.time_bin_start, 0.002, rtol=1e-12
    )


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_actual_two_dimensional_ahead_behind_uses_saved_decode_rows(family):
    tracking, spikes, marks, params = fixture(family)
    classifier, results = run_adapter(
        family, False, tracking, spikes, marks, params, intervals=[[0, 1]]
    )
    classifier.environments[0].track_graph = None
    time = tracking.index.to_numpy()
    position = pd.DataFrame(
        {
            "position_x": 100 * time,
            "position_y": np.zeros(len(time)),
            "head_orientation": np.zeros(len(time)),
        },
        index=time,
    )
    position.attrs.update(tracking.attrs)
    dummy = SimpleNamespace(
        fetch_model=lambda: classifier,
        fetch_results=lambda: results,
        fetch_position_info=lambda key: (
            position,
            ["position_x", "position_y"],
        ),
        fetch1=lambda key: {},
        get_orientation_col=lambda data: "head_orientation",
    )
    filename = "sorted_spikes.py" if family == "sorted" else "clusterless.py"
    name = (
        "SortedSpikesDecodingV1"
        if family == "sorted"
        else "ClusterlessDecodingV1"
    )
    method = source_method(filename, name, "get_ahead_behind_distance")
    mental = np.column_stack([100 * results.time.values, np.zeros(500)])
    with patch.object(
        analysis, "maximum_a_posteriori_estimate", return_value=mental
    ):
        distance = method(dummy)
    assert distance.shape == (500,)
    np.testing.assert_allclose(distance, 0.0, atol=1e-12)


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_actual_fetch_make_and_decode_keep_fractional_boundary_anchors(family):
    """Exercise the table fetch and make preparation before real decoding."""
    import copy

    tracking, spikes, marks, params = fixture(family)
    tracking["head_orientation"] = 0.0
    encoding = np.array([[0.05, 0.25]])
    decoding = np.array([[0.11, 0.31]])

    class PositionRelation:
        def __and__(self, key):
            return SimpleNamespace(fetch1_dataframe=lambda: tracking)

    class Part:
        def __and__(self, key):
            return self

        def fetch(self, name):
            return [0]

    core_fetch = source_method(
        "core.py", "PositionGroup", "fetch_position_info"
    )
    core_fetch.__globals__["PositionOutput"] = PositionRelation()

    class Group:
        Position = Part()

        def __and__(self, key):
            return self

        def fetch1(self, name):
            return {
                "KEY": {},
                "position_variables": ["position"],
                "upsample_rate": np.nan,
            }[name]

        def fetch_position_info(self, **kwargs):
            return core_fetch(self, **kwargs)

    exact, _ = core_fetch(Group(), min_time=0.05, max_time=0.31)
    np.testing.assert_array_equal(exact.index, tracking.index[2:10])
    assert exact.attrs["valid_position_intervals"] == [[0.05, 0.31]]

    class Relation:
        def __init__(self, values, key=None):
            self.values, self.key = values, key

        def __and__(self, key):
            return Relation(self.values, key)

        def fetch1(self, *args):
            if self.key is not None and "interval_list_name" in self.key:
                return self.values[self.key["interval_list_name"]]
            return self.values

    filename = "sorted_spikes.py" if family == "sorted" else "clusterless.py"
    name = (
        "SortedSpikesDecodingV1"
        if family == "sorted"
        else "ClusterlessDecodingV1"
    )
    fetch = source_method(filename, name, "fetch_position_info")
    fetch.__globals__.update(
        {
            "PositionGroup": Group(),
            "_get_interval_range": lambda key: (0.05, 0.31),
        }
    )
    table = SimpleNamespace(get_fully_defined_key=lambda key, **kwargs: key)
    table.fetch_position_info = lambda key: fetch(table, key)
    table.fetch_spike_data = lambda *args, **kwargs: (
        spikes if family == "sorted" else (spikes, marks)
    )
    captured = {}

    class Captured(Exception):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise Captured

    table._run_decoder = capture
    key = {
        "estimate_decoding_params": False,
        "decoding_param_name": "fixture",
        "nwb_file_name": "fixture.nwb",
        "position_group_name": "fixture",
        "encoding_interval": "encoding",
        "decoding_interval": "decoding",
    }
    preparation = source_method(
        filename, name, "make" if family == "sorted" else "make_fetch"
    )
    preparation.__globals__.update(
        {
            "copy": copy,
            "DecodingParameters": Relation(
                {"decoding_params": params, "decoding_kwargs": {}}
            ),
            "IntervalList": Relation(
                {"encoding": encoding, "decoding": decoding}
            ),
        }
    )
    if family == "sorted":
        with pytest.raises(Captured):
            preparation(table, key)
    else:
        values = preparation(table, key)
        captured = dict(
            zip(
                [
                    "decoding_params",
                    "decoding_kwargs",
                    "position_info",
                    "position_variable_names",
                    "spike_times",
                    "spike_waveform_features",
                    "encoding_interval",
                    "is_training",
                    "decoding_interval",
                ],
                values,
                strict=True,
            )
        )
    retained = captured["position_info"]
    expected_time = tracking.index.to_numpy()[1:11]
    np.testing.assert_array_equal(retained.index, expected_time)
    expected_training = (expected_time >= 0.05) & (expected_time <= 0.25)
    np.testing.assert_array_equal(
        captured["decoding_kwargs"]["is_training"], expected_training
    )
    assert not expected_training[[0, -1]].any()
    model, results = run_adapter(
        family,
        False,
        retained,
        spikes,
        marks,
        params,
        captured["decoding_kwargs"],
        decoding,
    )
    assert results.sizes["time"] == 100
    assert not results.is_missing.any()
    np.testing.assert_allclose(results.time[[0, -1]], [0.111, 0.309])

    # Zero-weight outside anchors must preserve the full original fit's exposure
    # and weighted rates, rather than extending an eligible endpoint hold.
    base = SortedSpikesDetector if family == "sorted" else ClusterlessDetector
    reference = base(**params)
    fit_kwargs = dict(
        position_time=tracking.index.to_numpy(),
        position=tracking[["position"]].to_numpy(),
        spike_times=spikes,
        is_training=(tracking.index >= 0.05) & (tracking.index <= 0.25),
        valid_position_intervals=[[0, 1]],
    )
    if family == "clusterless":
        fit_kwargs["spike_waveform_features"] = marks
    reference.fit(**fit_kwargs)
    for field in ["encoding_exposure_seconds", "mean_rates"]:
        np.testing.assert_allclose(
            model.encoding_model_[("", 0)][field],
            reference.encoding_model_[("", 0)][field],
            rtol=1e-6,
        )

    # Saved-result consumers also need the fetched bracketing samples.
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(100.0, 0.0))
    graph.add_edge(0, 1, distance=100.0)
    model.environments[0].track_graph = graph

    def linearize(**kwargs):
        np.testing.assert_array_equal(
            kwargs["position"], retained[["position"]]
        )
        return pd.DataFrame(
            {
                "projected_x_position": 100 * retained.index,
                "projected_y_position": 0.0,
                "track_segment_id": 0,
            }
        )

    table.fetch_environments = lambda key: model.environments
    linear_fetch = source_method(filename, name, "fetch_linear_position_info")
    linear_fetch.__globals__.update(
        {
            name: table,
            "PositionGroup": Group(),
            "get_linearized_position": linearize,
            "_get_interval_range": lambda key: (0.05, 0.31),
        }
    )
    linear = linear_fetch(table, key)
    np.testing.assert_array_equal(linear.index, retained.index)
    np.testing.assert_array_equal(
        utils.declared_tracking_intervals(linear),
        utils.declared_tracking_intervals(retained),
    )
    consumer = SimpleNamespace(
        fetch_model=lambda: model,
        fetch_results=lambda: results,
        fetch_linear_position_info=lambda key: linear,
        fetch1=lambda key: {},
        get_orientation_col=lambda data: "head_orientation",
    )
    method = source_method(filename, name, "get_ahead_behind_distance")
    mental = np.column_stack([100 * results.time.values, np.zeros(100)])
    with patch.object(
        analysis.distance1D,
        "_get_MAP_estimate_2d_position_edges",
        return_value=(mental, np.tile([0, 1], (100, 1))),
    ):
        distance = method(consumer)
    assert distance.shape == (100,)
    np.testing.assert_allclose(distance, 0.0, atol=1e-12)
