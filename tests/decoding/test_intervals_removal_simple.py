"""Unit tests for decoding utilities.

Tests the create_interval_labels, concatenate_interval_results, and
get_valid_kwargs functions from spyglass.decoding.v1.utils. These tests
validate the utility logic without requiring database infrastructure.

These are fast unit tests with no database dependencies.
"""

import logging

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


# ============================================================================
# Tests for get_valid_kwargs
# ============================================================================


@pytest.fixture
def get_valid_kwargs():
    """Import get_valid_kwargs inside fixture to defer database connection."""
    from spyglass.decoding.v1.utils import get_valid_kwargs

    return get_valid_kwargs


class MockClassifier:
    """Mock classifier for testing get_valid_kwargs."""

    def fit(self, position_time, position, spike_times, custom_fit_param=None):
        """Mock fit method with specific signature."""

    def predict(
        self,
        position_time,
        position,
        spike_times,
        time,
        custom_predict_param=None,
    ):
        """Mock predict method with specific signature."""


class TestGetValidKwargs:
    """Tests for the get_valid_kwargs function."""

    def test_valid_fit_kwargs_returned(self, get_valid_kwargs):
        """Valid fit kwargs should be returned in fit_kwargs dict."""
        classifier = MockClassifier()
        decoding_kwargs = {"custom_fit_param": 123}
        logger = logging.getLogger("test")

        fit_kwargs, predict_kwargs = get_valid_kwargs(
            classifier, decoding_kwargs, logger
        )

        assert fit_kwargs == {"custom_fit_param": 123}

    def test_valid_predict_kwargs_returned(self, get_valid_kwargs):
        """Valid predict kwargs should be returned in predict_kwargs dict."""
        classifier = MockClassifier()
        decoding_kwargs = {"custom_predict_param": "test"}
        logger = logging.getLogger("test")

        fit_kwargs, predict_kwargs = get_valid_kwargs(
            classifier, decoding_kwargs, logger
        )

        assert predict_kwargs == {"custom_predict_param": "test"}

    def test_kwargs_valid_for_both_returned_in_both(self, get_valid_kwargs):
        """Kwargs valid for both fit and predict should be in both dicts."""
        classifier = MockClassifier()
        # position_time is valid for both fit and predict
        decoding_kwargs = {"position_time": [1, 2, 3]}
        logger = logging.getLogger("test")

        fit_kwargs, predict_kwargs = get_valid_kwargs(
            classifier, decoding_kwargs, logger
        )

        assert "position_time" in fit_kwargs
        assert "position_time" in predict_kwargs

    def test_invalid_kwargs_triggers_warning(self, get_valid_kwargs, caplog):
        """Invalid kwargs should trigger a warning."""
        classifier = MockClassifier()
        decoding_kwargs = {"invalid_param": 123, "another_bad_one": "test"}
        logger = logging.getLogger("test")

        with caplog.at_level(logging.WARNING):
            get_valid_kwargs(classifier, decoding_kwargs, logger)

        assert (
            "not valid for classifier.fit or classifier.predict" in caplog.text
        )
        assert "invalid_param" in caplog.text
        assert "another_bad_one" in caplog.text

    def test_warning_includes_valid_kwargs(self, get_valid_kwargs, caplog):
        """Warning message should list valid kwargs for debugging."""
        classifier = MockClassifier()
        decoding_kwargs = {"invalid_param": 123}
        logger = logging.getLogger("test")

        with caplog.at_level(logging.WARNING):
            get_valid_kwargs(classifier, decoding_kwargs, logger)

        # Should include valid fit kwargs in the message
        assert "Valid fit kwargs" in caplog.text
        assert "custom_fit_param" in caplog.text
        # Should include valid predict kwargs in the message
        assert "Valid predict kwargs" in caplog.text
        assert "custom_predict_param" in caplog.text

    def test_empty_decoding_kwargs_no_warning(self, get_valid_kwargs, caplog):
        """Empty decoding_kwargs should not trigger any warning."""
        classifier = MockClassifier()
        decoding_kwargs = {}
        logger = logging.getLogger("test")

        with caplog.at_level(logging.WARNING):
            fit_kwargs, predict_kwargs = get_valid_kwargs(
                classifier, decoding_kwargs, logger
            )

        assert caplog.text == ""
        assert fit_kwargs == {}
        assert predict_kwargs == {}

    def test_all_valid_kwargs_no_warning(self, get_valid_kwargs, caplog):
        """When all kwargs are valid, no warning should be issued."""
        classifier = MockClassifier()
        decoding_kwargs = {
            "custom_fit_param": 123,
            "custom_predict_param": "test",
        }
        logger = logging.getLogger("test")

        with caplog.at_level(logging.WARNING):
            get_valid_kwargs(classifier, decoding_kwargs, logger)

        # No warning should be issued
        assert "not valid" not in caplog.text

    def test_mixed_valid_and_invalid_kwargs(self, get_valid_kwargs, caplog):
        """Mix of valid and invalid kwargs: valid returned, invalid warned."""
        classifier = MockClassifier()
        decoding_kwargs = {
            "custom_fit_param": 123,
            "invalid_param": "bad",
        }
        logger = logging.getLogger("test")

        with caplog.at_level(logging.WARNING):
            fit_kwargs, predict_kwargs = get_valid_kwargs(
                classifier, decoding_kwargs, logger
            )

        # Valid kwargs should be returned
        assert fit_kwargs == {"custom_fit_param": 123}
        # Invalid kwargs should trigger warning
        assert "invalid_param" in caplog.text
        # Valid kwargs should NOT be in the warning
        assert "custom_fit_param" not in caplog.text.split("will be ignored")[0]


# ============================================================================
# Tests for empty middle interval handling
# ============================================================================


class TestEmptyMiddleIntervalHandling:
    """Tests for handling empty intervals in the middle of a list.

    When an interval in the middle of a list is empty (no time points),
    it should be skipped and a warning should be logged. The interval_labels
    should not include a gap for the skipped interval.
    """

    def test_empty_middle_interval_skipped_in_labels(
        self, concatenate_interval_results
    ):
        """Empty intervals should be filtered before concatenation.

        When the real decoder code encounters an empty interval, it skips it
        and logs a warning. The resulting interval_labels should be consecutive
        integers without gaps.
        """
        # Simulate what the decoder does: filter out empty intervals before concat
        ds1 = _create_test_dataset(n_time=50, n_position=20, time_start=0.0)
        # ds2 would be empty - skipped in real code
        ds3 = _create_test_dataset(n_time=30, n_position=20, time_start=10.0)

        # After filtering, only non-empty intervals are passed
        result = concatenate_interval_results([ds1, ds3])

        labels = result.coords["interval_labels"].values
        # Labels should be 0, 1 (consecutive, no gap for skipped interval)
        np.testing.assert_array_equal(labels[:50], 0)
        np.testing.assert_array_equal(labels[50:], 1)

    def test_consecutive_labels_after_filtering(
        self, concatenate_interval_results
    ):
        """Labels should always be consecutive 0, 1, 2, ... after filtering."""
        ds1 = _create_test_dataset(n_time=20, n_position=10, time_start=0.0)
        ds2 = _create_test_dataset(n_time=15, n_position=10, time_start=5.0)
        ds3 = _create_test_dataset(n_time=25, n_position=10, time_start=10.0)

        result = concatenate_interval_results([ds1, ds2, ds3])

        labels = result.coords["interval_labels"].values
        unique_labels = np.unique(labels)

        # Should be consecutive integers starting from 0
        np.testing.assert_array_equal(unique_labels, [0, 1, 2])

    def test_single_remaining_interval_after_filtering(
        self, concatenate_interval_results
    ):
        """If only one interval remains after filtering, it should get label 0."""
        ds = _create_test_dataset(n_time=100, n_position=50)

        result = concatenate_interval_results([ds])

        labels = result.coords["interval_labels"].values
        assert np.all(labels == 0)
        assert len(labels) == 100


# ============================================================================
# Tests for interval_idx warning in DecodingOutput.create_decoding_view
# ============================================================================


class TestIntervalIdxWarning:
    """Tests for interval_idx warning when results lack interval_labels.

    These tests verify the warning behavior in DecodingOutput.create_decoding_view
    when interval_idx is specified but results don't have interval_labels.
    """

    def test_interval_idx_warning_when_no_labels(self, caplog):
        """Should warn when interval_idx specified but no interval_labels."""
        from unittest.mock import patch

        # Create results without interval_labels
        mock_results = xr.Dataset(
            {
                "acausal_posterior": (
                    ["time", "state_bins"],
                    np.random.rand(10, 5),
                )
            },
            coords={"time": np.arange(10), "state_bins": np.arange(5)},
        )

        from spyglass.decoding.decoding_merge import DecodingOutput

        with (
            patch.object(
                DecodingOutput, "fetch_results", return_value=mock_results
            ),
            caplog.at_level(logging.WARNING),
        ):
            try:
                DecodingOutput.create_decoding_view({}, interval_idx=0)
            except Exception:
                pass  # We only care about the warning, not visualization errors

        assert "interval_idx=0 specified but results do not" in caplog.text
        assert "interval_labels" in caplog.text
        assert "Ignoring interval_idx" in caplog.text

    def test_no_warning_when_interval_labels_present(self, caplog):
        """Should not warn when results have interval_labels."""
        from unittest.mock import patch

        # Create results WITH interval_labels
        mock_results = xr.Dataset(
            {
                "acausal_posterior": (
                    ["time", "state_bins"],
                    np.random.rand(10, 5),
                )
            },
            coords={
                "time": np.arange(10),
                "state_bins": np.arange(5),
                "interval_labels": ("time", np.array([0] * 5 + [1] * 5)),
            },
        )

        from spyglass.decoding.decoding_merge import DecodingOutput

        with (
            patch.object(
                DecodingOutput, "fetch_results", return_value=mock_results
            ),
            caplog.at_level(logging.WARNING),
        ):
            try:
                DecodingOutput.create_decoding_view({}, interval_idx=0)
            except Exception:
                pass  # We only care about the warning, not visualization errors

        # No warning about interval_idx should be issued
        assert "interval_idx" not in caplog.text

    def test_no_warning_when_interval_idx_is_none(self, caplog):
        """Should not warn when interval_idx is None."""
        from unittest.mock import patch

        # Create results without interval_labels
        mock_results = xr.Dataset(
            {
                "acausal_posterior": (
                    ["time", "state_bins"],
                    np.random.rand(10, 5),
                )
            },
            coords={"time": np.arange(10), "state_bins": np.arange(5)},
        )

        from spyglass.decoding.decoding_merge import DecodingOutput

        with (
            patch.object(
                DecodingOutput, "fetch_results", return_value=mock_results
            ),
            caplog.at_level(logging.WARNING),
        ):
            try:
                DecodingOutput.create_decoding_view({}, interval_idx=None)
            except Exception:
                pass  # We only care about the warning, not visualization errors

        # No warning should be issued when interval_idx is None
        assert "interval_idx" not in caplog.text
