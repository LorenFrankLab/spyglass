"""Unit tests for LinearizedPositionV1 with mocked external operations.

These tests use monkeypatch to replace expensive external operations (track_linearization)
with instant mock functions, allowing fast validation of Spyglass logic without waiting
for slow computations.

Runtime: ~1-2s (vs ~25s for integration tests)
"""

import pytest


def test_linearization_logic_mocked(
    sgpl,
    monkeypatch,
    mock_linearization,
    mock_linearization_save,
    lin_sel,
):
    """Test Spyglass linearization logic with mocked external operations.

    This test validates:
    - LinearizedPositionV1.populate() executes without errors
    - Table insertion succeeds with mocked data
    - All Spyglass code paths execute (fetch, params, intervals, database)
    - Result can be retrieved after population

    Mocked operations:
    - _compute_linearized_position() - Returns fake linearized position instantly
    - _save_linearization_results() - Returns fake object_id instantly

    Runtime: ~1-2s (vs ~25s for real linearization)
    """
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

    # Run populate - tests ALL Spyglass logic
    lin_v1_table = sgpl.LinearizedPositionV1()
    lin_v1_table.populate()

    # Verify results exist
    assert lin_v1_table, "LinearizedPositionV1 should have entries after populate"

    # Verify we can fetch results
    results = lin_v1_table.fetch()
    assert results is not None, "Should be able to fetch results"
    assert len(results) > 0, "Should have at least one result"

    # Verify key fields exist
    first_result = lin_v1_table.fetch(as_dict=True, limit=1)[0]
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
    assert first_result["linearized_position_object_id"] == "fake_linearized_position_object_id"

    print(f"✅ Mocked linearization test passed ({len(lin_v1_table)} entries)")
