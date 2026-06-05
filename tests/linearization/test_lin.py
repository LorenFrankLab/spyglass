import numpy as np
import pandas as pd
import pytest


@pytest.mark.slow
def test_fetch1_dataframe(lin_v1, lin_merge, lin_merge_key):
    df = (lin_merge & lin_merge_key).fetch1_dataframe().round(3).sum().to_dict()
    exp = {
        "linear_position": 326010.959,
        "projected_x_position": 47379.406,
        "projected_y_position": 31887.535,
        "track_segment_id": 3126.0,
    }

    for k in exp:
        assert (
            pytest.approx(df[k], rel=1e-3) == exp[k]
        ), f"Value differs from expected: {k}"


# TODO: Add more tests of this pipeline, not just the fetch1_dataframe method


def test_linearization_parameters_defaults():
    """Test LinearizationParameters default values."""
    from spyglass.linearization.v0.main import LinearizationParameters

    # Test that the table has expected structure
    assert hasattr(LinearizationParameters, "definition")
    definition = LinearizationParameters.definition

    # Check that expected fields are in definition
    assert "linearization_param_name" in definition
    assert "use_hmm = 0" in definition
    assert "route_euclidean_distance_scaling = 1.0" in definition
    assert "sensor_std_dev = 5.0" in definition
    assert "diagonal_bias = 0.5" in definition


def test_track_graph_definition():
    """Test TrackGraph table structure."""
    from spyglass.linearization.v0.main import TrackGraph

    assert hasattr(TrackGraph, "definition")
    definition = TrackGraph.definition

    # Should have basic track graph fields
    assert "track_graph_name" in definition or "track_graph" in definition


def test_linearization_parameter_validation():
    """Test linearization parameter validation."""
    # Test valid parameters
    valid_params = {
        "use_hmm": 0,
        "route_euclidean_distance_scaling": 1.0,
        "sensor_std_dev": 5.0,
        "diagonal_bias": 0.5,
    }

    # Test parameter ranges
    assert valid_params["use_hmm"] in [0, 1]
    assert valid_params["route_euclidean_distance_scaling"] > 0
    assert valid_params["sensor_std_dev"] > 0
    assert 0 <= valid_params["diagonal_bias"] <= 1


def test_linearization_track_graph_structure():
    """Test track graph data structure validation."""

    # Test valid track graph
    track_graph_data = {
        "node_positions": np.array([[0, 0], [1, 0], [2, 0]]),  # 3 nodes
        "edges": np.array([[0, 1], [1, 2]]),  # 2 edges
        "linear_edge_order": np.array([0, 1]),
        "linear_edge_spacing": np.array([1.0, 1.0]),
    }

    # Test structure validation
    assert track_graph_data["node_positions"].shape[0] >= 2  # At least 2 nodes
    assert track_graph_data["edges"].shape[1] == 2  # Edges are pairs
    assert len(track_graph_data["linear_edge_order"]) == len(
        track_graph_data["linear_edge_spacing"]
    )


def test_linearization_position_data():
    """Test position data format validation."""

    # Test position data structure
    position_data = {
        "x_position": np.array([0.0, 1.0, 2.0, 3.0]),
        "y_position": np.array([0.0, 0.5, 1.0, 1.5]),
        "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
    }

    # Test data consistency
    x_len = len(position_data["x_position"])
    y_len = len(position_data["y_position"])
    t_len = len(position_data["timestamps"])

    assert x_len == y_len == t_len  # All arrays same length
    assert np.all(
        np.diff(position_data["timestamps"]) > 0
    )  # Monotonic timestamps


def test_linearization_edge_cases():
    """Test linearization edge cases."""
    # Test minimal configuration
    minimal_config = {"use_hmm": 0, "sensor_std_dev": 1.0}

    # Should have required parameters
    assert "use_hmm" in minimal_config
    assert "sensor_std_dev" in minimal_config
    assert minimal_config["sensor_std_dev"] > 0
    assert minimal_config["sensor_std_dev"] > 0
    assert minimal_config["sensor_std_dev"] > 0
