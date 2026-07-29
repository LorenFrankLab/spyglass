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

    definition = TrackGraph.definition

    # Should have basic track graph fields
    assert "track_graph_name" in definition
    assert "node_positions" in definition
    assert "linear_edge_order" in definition
    assert "linear_edge_spacing" in definition
