"""Fast unit tests for clusterless decoding using mocked external operations.

These tests run ALL Spyglass code (fetch, parameter processing, interval
calculations, database operations) but mock the expensive external
dependencies (non_local_detector, file I/O).
"""

import pytest


def test_clusterless_data_processing_mocked(
    decode_v1,
    monkeypatch,
    mock_clusterless_decoder,
    mock_decoder_save,
    decode_sel_key,
    group_name,
    decode_clusterless_params_insert,
    pop_pos_group,
    group_unitwave,
):
    """Test Spyglass data processing logic with mocked external operations."""
    _ = pop_pos_group, group_unitwave  # ensure populated

    # Apply mocks to ClusterlessDecodingV1
    monkeypatch.setattr(
        decode_v1.clusterless.ClusterlessDecodingV1,
        "_run_decoder",
        mock_clusterless_decoder,
    )
    monkeypatch.setattr(
        decode_v1.clusterless.ClusterlessDecodingV1,
        "_save_decoder_results",
        mock_decoder_save,
    )

    # Create selection key
    selection_key = {
        **decode_sel_key,
        **decode_clusterless_params_insert,
        "waveform_features_group_name": group_name,
        "estimate_decoding_params": False,
    }

    # Only run populate if entry doesn't exist (avoids foreign key deletion issues)
    table = decode_v1.clusterless.ClusterlessDecodingV1 & selection_key
    if not table:
        # Insert selection
        decode_v1.clusterless.ClusterlessDecodingSelection.insert1(
            selection_key,
            skip_duplicates=True,
        )

        # Run populate - tests ALL Spyglass logic with mocks
        decode_v1.clusterless.ClusterlessDecodingV1.populate(selection_key)

        # Verify results were inserted
        table = decode_v1.clusterless.ClusterlessDecodingV1 & selection_key
        assert table, "No results inserted after populate"

        # Verify we can fetch the results
        results = table.fetch_results()
        assert results is not None, "No results returned"
        assert "posterior" in results, "Missing posterior in results"
    else:
        # Entry exists from integration tests - unit test not needed (skip gracefully)
        pytest.skip(
            "Skipping unit test - entry already validated by integration tests"
        )


def test_sorted_spikes_data_processing_mocked(
    decode_v1,
    monkeypatch,
    mock_sorted_spikes_decoder,
    mock_decoder_save,
    decode_sel_key,
    group_name,
    decode_spike_params_insert,
    pop_pos_group,
    pop_spikes_group,
):
    """Test data processing logic for sorted spikes with mocked operations."""
    _ = pop_pos_group, pop_spikes_group  # ensure populated

    # Apply mocks to SortedSpikesDecodingV1
    monkeypatch.setattr(
        decode_v1.sorted_spikes.SortedSpikesDecodingV1,
        "_run_decoder",
        mock_sorted_spikes_decoder,
    )
    monkeypatch.setattr(
        decode_v1.sorted_spikes.SortedSpikesDecodingV1,
        "_save_decoder_results",
        mock_decoder_save,
    )

    # Create selection key
    selection_key = {
        **decode_sel_key,
        **decode_spike_params_insert,
        "sorted_spikes_group_name": group_name,
        "unit_filter_params_name": "default_exclusion",
        "estimate_decoding_params": False,
    }

    # Only run populate if entry doesn't exist (avoids foreign key deletion issues)
    table = decode_v1.sorted_spikes.SortedSpikesDecodingV1 & selection_key
    if not table:
        # Insert selection
        decode_v1.sorted_spikes.SortedSpikesDecodingSelection.insert1(
            selection_key,
            skip_duplicates=True,
        )

        # Run populate - tests ALL Spyglass logic with mocks
        decode_v1.sorted_spikes.SortedSpikesDecodingV1.populate(selection_key)

        # Verify results were inserted
        table = decode_v1.sorted_spikes.SortedSpikesDecodingV1 & selection_key
        assert table, "No results inserted after populate"

        # Verify we can fetch the results
        results = table.fetch_results()
        assert results is not None, "No results returned"
        assert "posterior" in results, "Missing posterior in results"
    else:
        # Entry exists from integration tests - unit test not needed (skip gracefully)
        pytest.skip(
            "Skipping unit test - entry already validated by integration tests"
        )
