"""Tests for intervals dimension removal.

These tests verify that decoding results are stored as a single time series
with interval tracking, rather than using the intervals dimension which causes
padding and memory waste.
"""

import numpy as np
import pytest
import xarray as xr


def test_no_intervals_dimension_clusterless(
    decode_v1,
    monkeypatch,
    mock_clusterless_decoder,
    mock_decoder_save,
    decode_sel_key,
    group_name,
    decode_clusterless_params_insert,
    pop_pos_group,
    group_unitwave,
    mock_results_storage,
):
    """Test that clusterless decoding results don't use intervals dimension."""
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

    # Insert selection
    decode_v1.clusterless.ClusterlessDecodingSelection.insert1(
        selection_key,
        skip_duplicates=True,
    )

    # Run populate
    decode_v1.clusterless.ClusterlessDecodingV1.populate(selection_key)

    # Fetch results
    table = decode_v1.clusterless.ClusterlessDecodingV1 & selection_key
    results = table.fetch_results()

    # Verify that intervals is NOT a dimension
    assert "intervals" not in results.dims, (
        "Results should not have 'intervals' dimension - "
        "data should be concatenated along time instead"
    )

    # Verify that time is a dimension
    assert "time" in results.dims, "Results should have 'time' dimension"

    # Verify that interval_labels exists as a coordinate or variable
    assert (
        "interval_labels" in results.coords
        or "interval_labels" in results.data_vars
    ), (
        "Results should have 'interval_labels' to track which interval "
        "each time point belongs to"
    )


def test_interval_labels_tracking_clusterless(
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
    """Test that interval_labels correctly tracks intervals in clusterless decoding."""
    _ = pop_pos_group, group_unitwave  # ensure populated

    # Apply mocks
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

    # Insert and populate
    decode_v1.clusterless.ClusterlessDecodingSelection.insert1(
        selection_key,
        skip_duplicates=True,
    )
    decode_v1.clusterless.ClusterlessDecodingV1.populate(selection_key)

    # Fetch results
    table = decode_v1.clusterless.ClusterlessDecodingV1 & selection_key
    results = table.fetch_results()

    # Get interval_labels
    if "interval_labels" in results.coords:
        interval_labels = results.coords["interval_labels"]
    else:
        interval_labels = results["interval_labels"]

    # Verify interval_labels has same length as time
    assert len(interval_labels) == len(
        results.time
    ), "interval_labels should have same length as time dimension"

    # Verify interval_labels are integers starting from 0
    unique_labels = np.unique(interval_labels)
    assert np.all(
        unique_labels == np.arange(len(unique_labels))
    ), "interval_labels should be consecutive integers starting from 0"


def test_groupby_interval_labels_clusterless(
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
    """Test that results can be grouped by interval_labels."""
    _ = pop_pos_group, group_unitwave  # ensure populated

    # Apply mocks
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

    # Insert and populate
    decode_v1.clusterless.ClusterlessDecodingSelection.insert1(
        selection_key,
        skip_duplicates=True,
    )
    decode_v1.clusterless.ClusterlessDecodingV1.populate(selection_key)

    # Fetch results
    table = decode_v1.clusterless.ClusterlessDecodingV1 & selection_key
    results = table.fetch_results()

    # Test groupby operation
    grouped = results.groupby("interval_labels")

    # Verify groupby works
    assert grouped is not None, "Should be able to groupby interval_labels"

    # Verify we can iterate through groups
    for label, group in grouped:
        assert isinstance(
            label, (int, np.integer)
        ), "Group labels should be integers"
        assert "time" in group.dims, "Each group should have time dimension"


def test_no_intervals_dimension_sorted_spikes(
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
    """Test that sorted spikes decoding results don't use intervals dimension."""
    _ = pop_pos_group, pop_spikes_group  # ensure populated

    # Apply mocks
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

    # Insert and populate
    decode_v1.sorted_spikes.SortedSpikesDecodingSelection.insert1(
        selection_key,
        skip_duplicates=True,
    )
    decode_v1.sorted_spikes.SortedSpikesDecodingV1.populate(selection_key)

    # Fetch results
    table = decode_v1.sorted_spikes.SortedSpikesDecodingV1 & selection_key
    results = table.fetch_results()

    # Verify that intervals is NOT a dimension
    assert "intervals" not in results.dims, (
        "Results should not have 'intervals' dimension - "
        "data should be concatenated along time instead"
    )

    # Verify that time is a dimension
    assert "time" in results.dims, "Results should have 'time' dimension"

    # Verify that interval_labels exists
    assert (
        "interval_labels" in results.coords
        or "interval_labels" in results.data_vars
    ), (
        "Results should have 'interval_labels' to track which interval "
        "each time point belongs to"
    )
