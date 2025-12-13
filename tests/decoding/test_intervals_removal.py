"""Tests for intervals dimension removal.

These tests verify that decoding results are stored as a single time series
with interval tracking, rather than using the intervals dimension which causes
padding and memory waste.

Tests cover both estimate_decoding_params=False (predict branch) and
estimate_decoding_params=True (estimate_parameters branch).
"""

import numpy as np
import pytest


# ============================================================================
# Parametrized fixtures for decoder types
# ============================================================================


@pytest.fixture
def clusterless_decoder_config(
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
    """Configuration for clusterless decoder tests."""
    _ = pop_pos_group, group_unitwave  # ensure populated

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

    return {
        "selection_table": decode_v1.clusterless.ClusterlessDecodingSelection,
        "decoding_table": decode_v1.clusterless.ClusterlessDecodingV1,
        "base_key": {
            **decode_sel_key,
            **decode_clusterless_params_insert,
            "waveform_features_group_name": group_name,
        },
    }


@pytest.fixture
def sorted_spikes_decoder_config(
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
    """Configuration for sorted spikes decoder tests."""
    _ = pop_pos_group, pop_spikes_group  # ensure populated

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

    return {
        "selection_table": decode_v1.sorted_spikes.SortedSpikesDecodingSelection,
        "decoding_table": decode_v1.sorted_spikes.SortedSpikesDecodingV1,
        "base_key": {
            **decode_sel_key,
            **decode_spike_params_insert,
            "sorted_spikes_group_name": group_name,
            "unit_filter_params_name": "default_exclusion",
        },
    }


@pytest.fixture(
    params=["clusterless", "sorted_spikes"],
    ids=["clusterless", "sorted_spikes"],
)
def decoder_config(
    request, clusterless_decoder_config, sorted_spikes_decoder_config
):
    """Parametrized fixture providing both decoder configurations."""
    configs = {
        "clusterless": clusterless_decoder_config,
        "sorted_spikes": sorted_spikes_decoder_config,
    }
    return configs[request.param]


def _run_decoding(decoder_config, estimate_decoding_params):
    """Run decoding with given configuration and return results.

    Parameters
    ----------
    decoder_config : dict
        Configuration dictionary from decoder_config fixture containing:
        - selection_table: DataJoint table class for selection
        - decoding_table: DataJoint table class for decoding
        - base_key: Base key for database queries
    estimate_decoding_params : bool
        If True, estimate parameters over all time points.
        If False, predict on intervals only.

    Returns
    -------
    xr.Dataset
        Decoding results with interval_labels coordinate
    """
    selection_key = {
        **decoder_config["base_key"],
        "estimate_decoding_params": estimate_decoding_params,
    }

    decoder_config["selection_table"].insert1(
        selection_key, skip_duplicates=True
    )
    decoder_config["decoding_table"].populate(selection_key)

    table = decoder_config["decoding_table"] & selection_key
    return table.fetch_results()


# ============================================================================
# Tests for estimate_decoding_params=False (predict branch)
# ============================================================================


class TestNoIntervalsDimension:
    """Test that results use time dimension instead of intervals dimension."""

    def test_no_intervals_dimension(self, decoder_config):
        """Results should not have 'intervals' dimension."""
        results = _run_decoding(decoder_config, estimate_decoding_params=False)

        assert "intervals" not in results.dims, (
            "Results should not have 'intervals' dimension - "
            "data should be concatenated along time instead"
        )

    def test_has_time_dimension(self, decoder_config):
        """Results should have 'time' dimension."""
        results = _run_decoding(decoder_config, estimate_decoding_params=False)

        assert "time" in results.dims

    def test_has_interval_labels(self, decoder_config):
        """Results should have interval_labels coordinate."""
        results = _run_decoding(decoder_config, estimate_decoding_params=False)

        assert (
            "interval_labels" in results.coords
            or "interval_labels" in results.data_vars
        ), "Results should have 'interval_labels' to track interval membership"


class TestIntervalLabelsTracking:
    """Test that interval_labels correctly tracks intervals."""

    def test_labels_same_length_as_time(self, decoder_config):
        """interval_labels should have same length as time dimension."""
        results = _run_decoding(decoder_config, estimate_decoding_params=False)

        interval_labels = (
            results.coords["interval_labels"]
            if "interval_labels" in results.coords
            else results["interval_labels"]
        )

        assert len(interval_labels) == len(results.time)

    def test_labels_are_consecutive_integers(self, decoder_config):
        """interval_labels should be consecutive integers starting from 0."""
        results = _run_decoding(decoder_config, estimate_decoding_params=False)

        interval_labels = (
            results.coords["interval_labels"]
            if "interval_labels" in results.coords
            else results["interval_labels"]
        )

        unique_labels = np.unique(interval_labels)
        expected = np.arange(len(unique_labels))
        np.testing.assert_array_equal(unique_labels, expected)


class TestGroupbyOperation:
    """Test that results can be grouped by interval_labels."""

    def test_groupby_works(self, decoder_config):
        """Should be able to groupby interval_labels."""
        results = _run_decoding(decoder_config, estimate_decoding_params=False)

        grouped = results.groupby("interval_labels")
        assert grouped is not None

    def test_groups_have_time_dimension(self, decoder_config):
        """Each group should have time dimension."""
        results = _run_decoding(decoder_config, estimate_decoding_params=False)

        for label, group in results.groupby("interval_labels"):
            assert isinstance(label, (int, np.integer))
            assert "time" in group.dims


# ============================================================================
# Tests for estimate_decoding_params=True (estimate_parameters branch)
# ============================================================================


class TestEstimateParamsBranch:
    """Test interval_labels when estimate_decoding_params=True.

    When estimate_decoding_params=True, results span ALL time points and
    interval_labels should be:
    - -1 for time points outside any interval
    - 0, 1, 2, ... for time points inside intervals
    """

    def test_has_interval_labels(self, decoder_config):
        """Results should have interval_labels coordinate."""
        results = _run_decoding(decoder_config, estimate_decoding_params=True)

        assert "interval_labels" in results.coords

    def test_labels_same_length_as_time(self, decoder_config):
        """interval_labels should have same length as time dimension."""
        results = _run_decoding(decoder_config, estimate_decoding_params=True)

        interval_labels = results.coords["interval_labels"].values
        assert len(interval_labels) == len(results.time)

    def test_contains_negative_one_labels(self, decoder_config):
        """interval_labels should contain -1 for time points outside intervals."""
        results = _run_decoding(decoder_config, estimate_decoding_params=True)

        interval_labels = results.coords["interval_labels"].values
        assert (
            -1 in interval_labels
        ), "interval_labels should contain -1 for time points outside intervals"

    def test_contains_nonnegative_labels(self, decoder_config):
        """interval_labels should contain non-negative values for interval data."""
        results = _run_decoding(decoder_config, estimate_decoding_params=True)

        interval_labels = results.coords["interval_labels"].values
        assert np.any(
            interval_labels >= 0
        ), "interval_labels should contain non-negative values for interval data"

    def test_positive_labels_are_consecutive(self, decoder_config):
        """Positive interval_labels should be consecutive integers from 0."""
        results = _run_decoding(decoder_config, estimate_decoding_params=True)

        interval_labels = results.coords["interval_labels"].values
        positive_labels = interval_labels[interval_labels >= 0]
        unique_positive = np.unique(positive_labels)
        expected = np.arange(len(unique_positive))

        np.testing.assert_array_equal(unique_positive, expected)


class TestGroupbyWithNegativeLabels:
    """Test that groupby works correctly with -1 labels."""

    def test_groupby_includes_negative_one_group(self, decoder_config):
        """groupby should include a group for -1 (outside intervals)."""
        results = _run_decoding(decoder_config, estimate_decoding_params=True)

        labels_seen = [label for label, _ in results.groupby("interval_labels")]
        assert -1 in labels_seen

    def test_can_filter_to_interval_data_only(self, decoder_config):
        """Should be able to filter out -1 labels to get only interval data."""
        results = _run_decoding(decoder_config, estimate_decoding_params=True)

        interval_data = results.where(results.interval_labels >= 0, drop=True)

        assert len(interval_data.time) < len(results.time)
        assert np.all(interval_data.interval_labels >= 0)
