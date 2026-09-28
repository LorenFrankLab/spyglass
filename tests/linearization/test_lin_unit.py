"""Unit tests for LinearizedPositionV1 with mocked external operations.

These tests use monkeypatch to replace expensive external operations
(track_linearization) with instant mock functions, allowing fast validation of
Spyglass logic without waiting for slow computations.
"""


def test_linearization_logic_mocked(
    sgpl,
    monkeypatch,
    mock_linearization,
    mock_linearization_save,
    mock_lin_sel,
):
    """Test Spyglass linearization logic with mocked external operations.

    This test validates:
    - LinearizedPositionV1.populate() executes without errors
    - Table insertion succeeds with mocked data
    - All Spyglass code paths execute (fetch, params, intervals, database)
    - Result can be retrieved after population

    Mocked operations:
    - _compute_linearized_position() - Returns fake linearized position
    - _save_linearization_results() - Returns fake object_id
    """
    # Unpack the fixture
    lin_sel_table, mock_param_key = mock_lin_sel

    # Apply mocks to LinearizedPositionV1
    monkeypatch.setattr(
        sgpl.LinearizedPositionV1,
        "_compute_linearized_position",
        mock_linearization,
    )
    monkeypatch.setattr(
        sgpl.LinearizedPositionV1,
        "_save_linearization_results",
        mock_linearization_save,
    )

    # Run populate with restriction to only our mocked parameter set
    lin_v1_table = sgpl.LinearizedPositionV1()
    lin_v1_table.populate(mock_param_key)

    # Verify results exist for our specific parameter set
    restricted_table = lin_v1_table & mock_param_key
    assert (
        restricted_table
    ), "LinearizedPositionV1 should have entries after populate"

    # Verify we can fetch results (restricted to our parameter set)
    results = restricted_table.fetch()
    assert results is not None, "Should be able to fetch results"
    assert len(results) > 0, "Should have at least one result"

    # Verify key fields exist
    first_result = restricted_table.fetch(as_dict=True, limit=1)[0]
    expected_keys = [
        "pos_merge_id",
        "track_graph_name",
        "linearization_param_name",
        "analysis_file_name",
        "linearized_position_object_id",
    ]

    for key in expected_keys:
        assert key in first_result, f"Result should contain {key}"

    # Verify object_id is the mocked value
    assert (
        first_result["linearized_position_object_id"]
        == "fake_linearized_position_object_id"
    )
