"""Unit tests for intervals dimension removal utilities.

Tests the create_interval_labels and concatenate_interval_results functions
from spyglass.decoding.v1.utils. These tests validate the xarray concatenation
logic without requiring database infrastructure.

These are fast unit tests with no database dependencies.
"""

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def create_interval_labels():
    """Import create_interval_labels inside fixture to defer database connection."""
    from spyglass.decoding.v1.utils import create_interval_labels

    return create_interval_labels


@pytest.fixture
def concatenate_interval_results():
    """Import concatenate_interval_results inside fixture to defer database connection."""
    from spyglass.decoding.v1.utils import concatenate_interval_results

    return concatenate_interval_results


# ============================================================================
# Tests for create_interval_labels
# ============================================================================


class TestCreateIntervalLabels:
    """Tests for the create_interval_labels function."""

    def test_all_missing_returns_all_negative_one(self, create_interval_labels):
        """All time points marked as missing should get label -1."""
        is_missing = np.ones(100, dtype=bool)
        labels = create_interval_labels(is_missing)

        assert labels.shape == (100,)
        assert np.all(labels == -1)

    def test_single_contiguous_interval(self, create_interval_labels):
        """Single contiguous valid region gets label 0."""
        is_missing = np.ones(100, dtype=bool)
        is_missing[20:51] = False

        labels = create_interval_labels(is_missing)

        np.testing.assert_array_equal(labels[:20], -1)
        np.testing.assert_array_equal(labels[20:51], 0)
        np.testing.assert_array_equal(labels[51:], -1)

    def test_multiple_non_contiguous_intervals(self, create_interval_labels):
        """Multiple valid regions get sequential labels 0, 1, 2, ..."""
        is_missing = np.ones(100, dtype=bool)
        is_missing[10:21] = False  # interval 0
        is_missing[40:61] = False  # interval 1

        labels = create_interval_labels(is_missing)

        np.testing.assert_array_equal(labels[:10], -1)
        np.testing.assert_array_equal(labels[10:21], 0)
        np.testing.assert_array_equal(labels[21:40], -1)
        np.testing.assert_array_equal(labels[40:61], 1)
        np.testing.assert_array_equal(labels[61:], -1)

    def test_no_missing_data(self, create_interval_labels):
        """All valid data forms a single interval with label 0."""
        is_missing = np.zeros(50, dtype=bool)
        labels = create_interval_labels(is_missing)

        assert np.all(labels == 0)

    def test_single_point_intervals(self, create_interval_labels):
        """Single-point valid regions should each get their own label."""
        is_missing = np.ones(10, dtype=bool)
        is_missing[2] = False
        is_missing[5] = False
        is_missing[8] = False

        labels = create_interval_labels(is_missing)

        assert labels[2] == 0
        assert labels[5] == 1
        assert labels[8] == 2
        assert np.sum(labels >= 0) == 3

    def test_returns_correct_dtype(self, create_interval_labels):
        """Labels should be integer type."""
        is_missing = np.zeros(10, dtype=bool)
        labels = create_interval_labels(is_missing)

        assert np.issubdtype(labels.dtype, np.integer)


# ============================================================================
# Tests for concatenate_interval_results
# ============================================================================


def _create_test_dataset(n_time, n_position, time_start=0.0, time_step=0.1):
    """Helper to create a test xarray Dataset mimicking decoding results."""
    time = np.arange(time_start, time_start + n_time * time_step, time_step)[
        :n_time
    ]
    return xr.Dataset(
        {
            "posterior": (
                ["time", "position"],
                np.random.rand(n_time, n_position),
            )
        },
        coords={"time": time, "position": np.arange(n_position)},
    )


class TestConcatenateIntervalResults:
    """Tests for the concatenate_interval_results function."""

    def test_empty_list_raises_valueerror(self, concatenate_interval_results):
        """Empty input list should raise ValueError."""
        with pytest.raises(
            ValueError, match="All decoding intervals are empty"
        ):
            concatenate_interval_results([])

    def test_single_interval(self, concatenate_interval_results):
        """Single interval should work and get label 0."""
        dataset = _create_test_dataset(n_time=50, n_position=20)
        result = concatenate_interval_results([dataset])

        assert "interval_labels" in result.coords
        assert len(result.time) == 50
        assert np.all(result.coords["interval_labels"].values == 0)

    def test_multiple_intervals_concatenated_along_time(
        self, concatenate_interval_results
    ):
        """Multiple intervals should concatenate along time dimension."""
        ds1 = _create_test_dataset(n_time=100, n_position=50, time_start=0.0)
        ds2 = _create_test_dataset(n_time=70, n_position=50, time_start=15.0)

        result = concatenate_interval_results([ds1, ds2])

        assert "intervals" not in result.dims
        assert "time" in result.dims
        assert len(result.time) == 170
        assert result.posterior.shape == (170, 50)

    def test_interval_labels_track_source_intervals(
        self, concatenate_interval_results
    ):
        """interval_labels should correctly identify source interval."""
        ds1 = _create_test_dataset(n_time=100, n_position=50)
        ds2 = _create_test_dataset(n_time=70, n_position=50, time_start=15.0)
        ds3 = _create_test_dataset(n_time=30, n_position=50, time_start=25.0)

        result = concatenate_interval_results([ds1, ds2, ds3])

        labels = result.coords["interval_labels"].values
        assert len(labels) == 200
        np.testing.assert_array_equal(labels[:100], 0)
        np.testing.assert_array_equal(labels[100:170], 1)
        np.testing.assert_array_equal(labels[170:], 2)

    def test_groupby_works_on_result(self, concatenate_interval_results):
        """Result should support groupby on interval_labels."""
        ds1 = _create_test_dataset(n_time=100, n_position=50)
        ds2 = _create_test_dataset(n_time=70, n_position=50, time_start=15.0)

        result = concatenate_interval_results([ds1, ds2])
        grouped = result.groupby("interval_labels")

        group_sizes = {label: len(group.time) for label, group in grouped}
        assert group_sizes[0] == 100
        assert group_sizes[1] == 70

    def test_no_padding_waste(self, concatenate_interval_results):
        """Concatenation should not introduce padding (memory efficiency)."""
        ds1 = _create_test_dataset(n_time=100, n_position=50)
        ds2 = _create_test_dataset(n_time=70, n_position=50, time_start=15.0)

        result = concatenate_interval_results([ds1, ds2])

        # Total elements should be exactly sum of inputs, not max*n_intervals
        expected_elements = (100 * 50) + (70 * 50)
        actual_elements = result.posterior.size
        assert actual_elements == expected_elements

    def test_empty_dataset_in_list_raises_error(
        self, concatenate_interval_results
    ):
        """Dataset with empty time dimension should raise ValueError."""
        ds1 = _create_test_dataset(n_time=50, n_position=20)
        ds_empty = xr.Dataset(
            {"posterior": (["time", "position"], np.empty((0, 20)))},
            coords={"time": [], "position": np.arange(20)},
        )

        with pytest.raises(
            ValueError, match="Interval 1 has empty time dimension"
        ):
            concatenate_interval_results([ds1, ds_empty])

    def test_preserves_other_coordinates(self, concatenate_interval_results):
        """Non-time coordinates should be preserved."""
        ds = xr.Dataset(
            {
                "posterior": (["time", "state_bins"], np.random.rand(50, 100)),
            },
            coords={
                "time": np.arange(50),
                "state_bins": np.arange(100),
                "states": ("states", ["Continuous", "Fragmented"]),
            },
        )

        result = concatenate_interval_results([ds])

        assert "states" in result.coords
        assert list(result.coords["states"].values) == [
            "Continuous",
            "Fragmented",
        ]

    def test_preserves_data_variables(self, concatenate_interval_results):
        """All data variables should be preserved."""
        ds = xr.Dataset(
            {
                "acausal_posterior": (
                    ["time", "position"],
                    np.random.rand(50, 20),
                ),
                "causal_posterior": (
                    ["time", "position"],
                    np.random.rand(50, 20),
                ),
            },
            coords={"time": np.arange(50), "position": np.arange(20)},
        )

        result = concatenate_interval_results([ds])

        assert "acausal_posterior" in result.data_vars
        assert "causal_posterior" in result.data_vars
