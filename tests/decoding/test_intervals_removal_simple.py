"""Unit tests for intervals dimension removal utilities.

Tests the create_interval_labels and concatenate_interval_results functions
from spyglass.decoding.v1.utils. These tests validate the xarray concatenation
logic without requiring database infrastructure.
"""

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def interval_utils():
    """Import utils module inside fixture to defer database connection."""
    from spyglass.decoding.v1.utils import (
        concatenate_interval_results,
        create_interval_labels,
    )

    return create_interval_labels, concatenate_interval_results


def test_concatenation_without_intervals_dimension(interval_utils):
    """Test that concatenating along time instead of intervals works correctly."""
    create_interval_labels, concatenate_interval_results = interval_utils

    # Simulate two intervals with different lengths (like real decoding)
    interval1_time = np.arange(0, 10, 0.1)
    interval2_time = np.arange(15, 22, 0.1)
    n_position = 50

    # Create results for each interval
    results = []
    interval_labels = []

    for interval_idx, interval_time in enumerate(
        [interval1_time, interval2_time]
    ):
        n_time = len(interval_time)
        interval_result = xr.Dataset(
            {
                "posterior": (
                    ["time", "position"],
                    np.random.rand(n_time, n_position),
                ),
            },
            coords={"time": interval_time, "position": np.arange(n_position)},
        )
        results.append(interval_result)
        interval_labels.extend([interval_idx] * n_time)

    # Concatenate along time dimension (new approach)
    concatenated = xr.concat(results, dim="time")
    concatenated = concatenated.assign_coords(
        interval_labels=("time", interval_labels)
    )

    # Verify structure
    assert (
        "intervals" not in concatenated.dims
    ), "Should not have intervals dimension"
    assert "time" in concatenated.dims, "Should have time dimension"
    assert (
        "interval_labels" in concatenated.coords
    ), "Should have interval_labels coordinate"

    # Verify shape - should be (total_time, n_position) not (n_intervals, max_time, n_position)
    expected_time_points = len(interval1_time) + len(interval2_time)
    assert concatenated.posterior.shape == (
        expected_time_points,
        n_position,
    ), f"Expected shape ({expected_time_points}, {n_position}), got {concatenated.posterior.shape}"

    # Verify interval_labels
    assert (
        len(concatenated.coords["interval_labels"]) == expected_time_points
    ), "interval_labels should have same length as time"
    unique_labels = np.unique(concatenated.coords["interval_labels"].values)
    assert len(unique_labels) == 2, "Should have 2 unique interval labels"
    assert list(unique_labels) == [0, 1], "Interval labels should be [0, 1]"

    # Verify groupby works
    grouped = concatenated.groupby("interval_labels")
    group_sizes = {label: len(group.time) for label, group in grouped}
    assert group_sizes[0] == len(
        interval1_time
    ), f"Interval 0 should have {len(interval1_time)} time points"
    assert group_sizes[1] == len(
        interval2_time
    ), f"Interval 1 should have {len(interval2_time)} time points"


def test_memory_efficiency(interval_utils):
    """Test that new approach uses less memory than intervals dimension."""
    create_interval_labels, concatenate_interval_results = interval_utils

    # Create two intervals with different lengths
    interval1_data = xr.Dataset(
        {
            "posterior": (["time", "position"], np.random.rand(100, 50)),
        },
        coords={"time": np.arange(100), "position": np.arange(50)},
    )

    interval2_data = xr.Dataset(
        {
            "posterior": (["time", "position"], np.random.rand(70, 50)),
        },
        coords={"time": np.arange(70), "position": np.arange(50)},
    )

    # Old approach: concat with intervals dimension (creates padding)
    old_approach = xr.concat([interval1_data, interval2_data], dim="intervals")

    # New approach: concat along time
    interval2_shifted = interval2_data.assign_coords(
        time=interval2_data.time + 100
    )
    new_approach = xr.concat([interval1_data, interval2_shifted], dim="time")
    new_approach = new_approach.assign_coords(
        interval_labels=("time", [0] * 100 + [1] * 70)
    )

    # Old approach has shape (2, 100, 50) = 10000 values with 1500 padding zeros
    # New approach has shape (170, 50) = 8500 values with no padding
    assert (
        old_approach.posterior.size > new_approach.posterior.size
    ), "Old approach should use more memory due to padding"

    # Calculate memory savings
    old_bytes = old_approach.nbytes
    new_bytes = new_approach.nbytes
    savings_percent = ((old_bytes - new_bytes) / old_bytes) * 100

    assert savings_percent > 0, "New approach should save memory"


def test_empty_intervals_raises_error(interval_utils):
    """Test that concatenate_interval_results raises ValueError for empty list.

    This ensures proper error handling when all decoding intervals are empty
    (e.g., intervals don't overlap with position data).
    """
    create_interval_labels, concatenate_interval_results = interval_utils

    with pytest.raises(ValueError, match="All decoding intervals are empty"):
        concatenate_interval_results([])


def test_create_interval_labels_all_missing(interval_utils):
    """Test create_interval_labels when all data is missing."""
    create_interval_labels, concatenate_interval_results = interval_utils

    # All time points marked as missing
    is_missing = np.ones(100, dtype=bool)
    labels = create_interval_labels(is_missing)

    # All labels should be -1 (outside any interval)
    assert np.all(
        labels == -1
    ), "All labels should be -1 when all data is missing"


def test_create_interval_labels_single_interval(interval_utils):
    """Test create_interval_labels with a single contiguous interval."""
    create_interval_labels, concatenate_interval_results = interval_utils

    # Single interval from index 20-50
    is_missing = np.ones(100, dtype=bool)
    is_missing[20:51] = False

    labels = create_interval_labels(is_missing)

    # Check -1 for outside, 0 for inside
    assert np.all(labels[:20] == -1), "Labels before interval should be -1"
    assert np.all(labels[20:51] == 0), "Labels inside interval should be 0"
    assert np.all(labels[51:] == -1), "Labels after interval should be -1"


def test_create_interval_labels_multiple_intervals(interval_utils):
    """Test create_interval_labels with multiple non-contiguous intervals."""
    create_interval_labels, concatenate_interval_results = interval_utils

    # Two intervals: 10-20 and 40-60
    is_missing = np.ones(100, dtype=bool)
    is_missing[10:21] = False
    is_missing[40:61] = False

    labels = create_interval_labels(is_missing)

    # Check labels
    assert np.all(
        labels[:10] == -1
    ), "Labels before first interval should be -1"
    assert np.all(labels[10:21] == 0), "First interval should have label 0"
    assert np.all(labels[21:40] == -1), "Gap between intervals should be -1"
    assert np.all(labels[40:61] == 1), "Second interval should have label 1"
    assert np.all(
        labels[61:] == -1
    ), "Labels after second interval should be -1"
