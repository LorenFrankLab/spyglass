"""Tests for intervals dimension removal.

These tests verify that decoding results are stored as a single time series
with interval tracking, rather than using the intervals dimension which causes
padding and memory waste.

Tests cover both estimate_decoding_params=False (predict branch) and
estimate_decoding_params=True (estimate_parameters branch).
"""

import numpy as np


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


# ============================================================================
# Tests for estimate_decoding_params=True branch
# ============================================================================


def test_interval_labels_estimate_params_clusterless(
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
    """Test interval_labels when estimate_decoding_params=True (clusterless).

    When estimate_decoding_params=True, results span ALL time points and
    interval_labels should be:
    - -1 for time points outside any interval
    - 0, 1, 2, ... for time points inside intervals
    """
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

    # Create selection key with estimate_decoding_params=True
    selection_key = {
        **decode_sel_key,
        **decode_clusterless_params_insert,
        "waveform_features_group_name": group_name,
        "estimate_decoding_params": True,
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

    # Verify interval_labels exists
    assert "interval_labels" in results.coords, (
        "Results should have 'interval_labels' coordinate when "
        "estimate_decoding_params=True"
    )

    # Get interval_labels
    interval_labels = results.coords["interval_labels"].values

    # Verify interval_labels has same length as time
    assert len(interval_labels) == len(
        results.time
    ), "interval_labels should have same length as time dimension"

    # Verify that -1 exists (for times outside intervals)
    assert -1 in interval_labels, (
        "interval_labels should contain -1 for time points outside intervals "
        "when estimate_decoding_params=True"
    )

    # Verify that non-negative labels exist (for times inside intervals)
    assert np.any(interval_labels >= 0), (
        "interval_labels should contain non-negative values for time points "
        "inside intervals"
    )

    # Verify labels are consecutive integers starting from 0 (excluding -1)
    positive_labels = interval_labels[interval_labels >= 0]
    unique_positive = np.unique(positive_labels)
    expected_labels = np.arange(len(unique_positive))
    np.testing.assert_array_equal(
        unique_positive,
        expected_labels,
        err_msg="Positive interval_labels should be consecutive integers from 0",
    )


def test_interval_labels_estimate_params_sorted_spikes(
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
    """Test interval_labels when estimate_decoding_params=True (sorted spikes).

    When estimate_decoding_params=True, results span ALL time points and
    interval_labels should be:
    - -1 for time points outside any interval
    - 0, 1, 2, ... for time points inside intervals
    """
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

    # Create selection key with estimate_decoding_params=True
    selection_key = {
        **decode_sel_key,
        **decode_spike_params_insert,
        "sorted_spikes_group_name": group_name,
        "unit_filter_params_name": "default_exclusion",
        "estimate_decoding_params": True,
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

    # Verify interval_labels exists
    assert "interval_labels" in results.coords, (
        "Results should have 'interval_labels' coordinate when "
        "estimate_decoding_params=True"
    )

    # Get interval_labels
    interval_labels = results.coords["interval_labels"].values

    # Verify interval_labels has same length as time
    assert len(interval_labels) == len(
        results.time
    ), "interval_labels should have same length as time dimension"

    # Verify that -1 exists (for times outside intervals)
    assert -1 in interval_labels, (
        "interval_labels should contain -1 for time points outside intervals "
        "when estimate_decoding_params=True"
    )

    # Verify that non-negative labels exist (for times inside intervals)
    assert np.any(interval_labels >= 0), (
        "interval_labels should contain non-negative values for time points "
        "inside intervals"
    )


def test_groupby_works_with_negative_labels(
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
    """Test that groupby works correctly with -1 labels.

    Users should be able to:
    - Group by interval_labels to iterate through intervals
    - Filter out -1 labels to get only interval data
    """
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

    # Create selection key with estimate_decoding_params=True
    selection_key = {
        **decode_sel_key,
        **decode_clusterless_params_insert,
        "waveform_features_group_name": group_name,
        "estimate_decoding_params": True,
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

    # Test groupby operation works
    grouped = results.groupby("interval_labels")
    assert grouped is not None, "Should be able to groupby interval_labels"

    # Verify we can iterate and get the -1 group
    labels_seen = []
    for label, group in grouped:
        labels_seen.append(label)
        assert "time" in group.dims, "Each group should have time dimension"

    # Verify -1 is one of the groups
    assert -1 in labels_seen, "Should have a group for -1 (outside intervals)"

    # Test filtering to only interval data
    interval_data = results.where(results.interval_labels >= 0, drop=True)
    assert len(interval_data.time) < len(
        results.time
    ), "Filtering to interval_labels >= 0 should reduce data size"
    assert np.all(
        interval_data.interval_labels >= 0
    ), "Filtered data should only have non-negative interval_labels"
