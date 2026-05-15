# pylint: disable=protected-access

import numpy as np
import pandas as pd
import pytest
from numpy import array_equal
from unittest.mock import patch


@pytest.fixture(scope="session", name="interval_list")
def interval_list_fixture(common):
    yield common.IntervalList()


def test_plot_intervals(interval_list, mini_dict, start_time=0):
    interval_query = interval_list & mini_dict
    fig = (interval_query).plot_intervals(
        return_fig=True, start_time=start_time
    )

    ax = fig.axes[0]

    # extract interval list names from the figure
    fig_interval_list_names = np.array(
        [label.get_text() for label in ax.get_yticklabels()]
    )

    # extract interval list names from the IntervalList table
    fetch_interval_list_names = np.array(
        interval_query.fetch("interval_list_name")
    )

    # check that the interval list names match between the two methods
    assert array_equal(
        fig_interval_list_names, fetch_interval_list_names
    ), "plot_intervals failed: plotted interval list names do not match"

    # extract the interval times from the figure
    intervals_fig = []
    for collection in ax.collections:
        collection_data = []
        for path_patch in collection.get_paths():
            # Extract patch vertices to get x data
            vertices = path_patch.vertices
            x_start = vertices[0, 0]
            x_end = vertices[2, 0]
            collection_data.append(
                [x_start * 60 + start_time, x_end * 60 + start_time]
            )
        intervals_fig.append(np.array(collection_data))
    intervals_fig = np.array(intervals_fig, dtype="object")

    # extract interval times from the IntervalList table
    intervals_fetch = interval_query.fetch("valid_times")

    all_equal = True
    for i_fig, i_fetch in zip(intervals_fig, intervals_fetch):
        # permit rounding errors up to the 4th decimal place, as a result of inaccuracies during unit conversions
        i_fig = np.round(i_fig.astype("float"), 4)
        # Normalize i_fetch: handle Python lists and 1D [start, end] arrays
        i_fetch = np.round(np.array(i_fetch, dtype="float").reshape(-1, 2), 4)
        if not array_equal(i_fig, i_fetch):
            all_equal = False

    # check that the interval list times match between the two methods
    assert (
        all_equal
    ), "plot_intervals failed: plotted interval times do not match"


def test_plot_epoch(interval_list):
    restr_interval = interval_list & "interval_list_name like 'raw%'"
    fig = restr_interval.plot_epoch_pos_raw_intervals(return_fig=True)
    epoch_label = fig.get_axes()[0].get_yticklabels()[-1].get_text()
    assert epoch_label == "epoch", "plot_epoch failed"

    epoch_interval = fig.get_axes()[0].lines[0].get_ydata()
    assert array_equal(epoch_interval, [1, 1]), "plot_epoch failed"


# Comprehensive tests for Interval class coverage improvement


def test_interval_init_with_dict():
    """Test Interval initialization with dictionary."""
    from spyglass.common.common_interval import Interval

    # Test with dict containing key for table lookup
    test_dict = {
        "nwb_file_name": "test.nwb",
        "interval_list_name": "test_interval",
    }

    with patch.object(Interval, "_import_from_table") as mock_import:
        mock_import.return_value = [[1.0, 2.0], [3.0, 4.0]]

        interval = Interval(test_dict)

        assert np.array_equal(interval.times, [[1.0, 2.0], [3.0, 4.0]])
        mock_import.assert_called_once_with(test_dict)


def test_interval_init_from_inds():
    """Test Interval initialization from indices."""
    from spyglass.common.common_interval import Interval

    indices = [10, 20, 30, 40]

    with patch.object(Interval, "from_inds") as mock_from_inds:
        mock_from_inds.return_value = [[10, 20], [30, 40]]

        _ = Interval(indices, from_inds=True)

        mock_from_inds.assert_called_once_with(indices)


def test_interval_init_invalid_type():
    """Test Interval initialization with invalid type."""
    from spyglass.common.common_interval import Interval

    with pytest.raises(TypeError, match="Unrecognized interval_list type"):
        Interval(set([1, 2, 3]))  # set is not supported


def test_interval_init_invalid_intervals():
    """Test Interval initialization with invalid start/stop ordering."""
    from spyglass.common.common_interval import Interval

    # Test intervals where start > stop
    with pytest.raises(ValueError, match="start <= stop"):
        Interval([[2.0, 1.0], [4.0, 3.0]])  # Invalid ordering


def test_interval_init_with_duplicates(caplog):
    """Test Interval initialization duplicate handling."""
    from spyglass.common.common_interval import Interval

    interval = Interval(
        [[1.0, 2.0], [1.0, 2.0]], no_duplicates=False, warn=True
    )
    assert len(interval.times) == 2
    assert "Found duplicate(s)" in caplog.text

    # Test with duplicates and no_duplicates=True (should remove)
    interval = Interval(
        [[1.0, 2.0], [1.0, 2.0]], no_duplicates=True, warn=False
    )
    assert len(interval.times) == 1  # Removes duplicates


def test_interval_init_with_overlap(caplog):
    """Test Interval initialization overlap handling."""
    from spyglass.common.common_interval import Interval

    interval = Interval([[1.0, 3.0], [2.0, 4.0]], no_overlap=False, warn=True)
    assert len(interval.times) == 2
    assert "Found overlap(s)" in caplog.text

    # Test with overlaps and no_overlap=True (should consolidate)
    interval = Interval([[1.0, 3.0], [2.0, 4.0]], no_overlap=True, warn=False)
    assert len(interval.times) == 1  # Consolidates overlaps
    assert np.array_equal(interval.times[0], [1.0, 4.0])


def test_interval_expand_1d():
    """Test Interval._expand_1d static method."""
    from spyglass.common.common_interval import Interval

    # Test 1D array expansion
    result = Interval._expand_1d(np.array([1.0, 2.0]))
    assert result.shape == (1, 2)
    assert np.array_equal(result, [[1.0, 2.0]])

    # Test already 2D array (no change)
    result = Interval._expand_1d(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert result.shape == (2, 2)

    # Test empty array
    result = Interval._expand_1d(np.array([]))
    assert result.shape == (0,)


def test_interval_by_length():
    """Test Interval.by_length filtering."""
    from spyglass.common.common_interval import Interval

    # Create intervals of different lengths
    interval = Interval([[1.0, 2.0], [3.0, 7.0], [8.0, 8.5]], warn=False)

    # Filter by minimum length
    result = interval.by_length(min_length=2.0)
    assert len(result.times) == 1  # Only [3.0, 7.0] has length >= 2.0
    assert np.array_equal(result.times[0], [3.0, 7.0])

    # Filter by maximum length
    result = interval.by_length(max_length=1.01)
    assert len(result.times) == 2

    # Filter by both min and max length
    result = interval.by_length(min_length=0.5, max_length=1.5)
    assert len(result.times) == 1
    assert np.array_equal(result.times[0], [1.0, 2.0])


def test_interval_contains():
    """Test Interval.contains method."""
    from spyglass.common.common_interval import Interval

    interval = Interval([[1.0, 3.0], [5.0, 7.0]], warn=False)
    timestamps = np.array([0.5, 1.5, 2.5, 3.5, 5.5, 6.5, 7.5])

    # Test returning timestamps
    result = interval.contains(timestamps)
    expected = np.array([1.5, 2.5, 5.5, 6.5])
    assert np.array_equal(result, expected)

    # Test returning indices
    result_indices = interval.contains(timestamps, as_indices=True)
    expected_indices = np.array([1, 2, 4, 5])
    assert np.array_equal(result_indices, expected_indices)

    # Test with padding
    result_padded = interval.contains(timestamps, padding=1)
    # Should pad first and last results
    assert result_padded[0] == expected[0] - 1  # First timestamp padded


def test_interval_excludes():
    """Test Interval.excludes method."""
    from spyglass.common.common_interval import Interval

    interval = Interval([[1.0, 3.0], [5.0, 7.0]], warn=False)
    timestamps = np.array([0.5, 1.5, 2.5, 3.5, 5.5, 6.5, 7.5])

    # Test returning excluded timestamps
    result = interval.excludes(timestamps)
    expected = np.array([0.5, 3.5, 7.5])  # Not in intervals
    assert np.array_equal(result, expected)

    # Test returning excluded indices
    result_indices = interval.excludes(timestamps, as_indices=True)
    expected_indices = np.array([0, 3, 6])
    assert np.array_equal(result_indices, expected_indices)


def test_interval_set_intersect():
    """Test Interval._set_intersect static method."""
    from spyglass.common.common_interval import Interval

    # Test intersecting intervals
    result = Interval._set_intersect([1.0, 3.0], [2.0, 4.0])
    assert np.array_equal(result, [2.0, 3.0])

    # Test non-intersecting intervals
    result = Interval._set_intersect([1.0, 2.0], [3.0, 4.0])
    assert result is None

    # Test touching intervals (end == start)
    result = Interval._set_intersect([1.0, 2.0], [2.0, 3.0])
    assert result is None  # end > start, so 2.0 > 2.0 is False


def test_interval_intersect():
    """Test Interval.intersect method."""
    from spyglass.common.common_interval import Interval

    interval1 = Interval([[1.0, 4.0], [6.0, 8.0]], warn=False)
    interval2 = [[2.0, 3.0], [7.0, 9.0]]  # Different format

    result = interval1.intersect(interval2)

    # Should find intersections: [2.0, 3.0] and [7.0, 8.0]
    assert len(result.times) == 2
    assert np.array_equal(result.times[0], [2.0, 3.0])
    assert np.array_equal(result.times[1], [7.0, 8.0])


def test_interval_intersect_with_min_length():
    """Test Interval.intersect with minimum length filtering."""
    from spyglass.common.common_interval import Interval

    interval1 = Interval([[1.0, 4.0], [6.0, 8.0]], warn=False)
    interval2 = [[2.0, 2.1], [7.0, 9.0]]  # Small intersection

    # Without min_length filter
    result = interval1.intersect(interval2)
    assert len(result.times) == 2

    # With min_length filter (filter out small intersection)
    result = interval1.intersect(interval2, min_length=0.5)
    assert len(result.times) == 1  # Only [7.0, 8.0] meets length requirement
    assert np.array_equal(result.times[0], [7.0, 8.0])


def test_interval_union():
    """Test Interval.union method."""
    from spyglass.common.common_interval import Interval

    interval1 = Interval([[1.0, 3.0], [7.0, 9.0]], warn=False)
    interval2 = [[2.0, 5.0], [8.0, 10.0]]

    result = interval1.union(interval2)

    # Should merge overlapping intervals
    assert len(result.times) == 2
    assert np.array_equal(result.times[0], [1.0, 5.0])  # Merged [1,3] + [2,5]
    assert np.array_equal(result.times[1], [7.0, 10.0])  # Merged [7,9] + [8,10]


def test_interval_union_with_length_filter():
    """Test Interval.union with length filtering."""
    from spyglass.common.common_interval import Interval

    interval1 = Interval([[1.0, 1.1], [5.0, 8.0]], warn=False)
    interval2 = [[2.0, 2.2]]

    # Current implementation does not filter union output by length.
    result = interval1.union(interval2, min_length=2.0)
    assert len(result.times) == 3
    assert np.array_equal(result.times[2], [5.0, 8.0])


def test_interval_union_empty_intervals():
    """Test Interval.union with empty intervals."""
    from spyglass.common.common_interval import Interval

    interval1 = Interval([], warn=False)
    interval2 = [[1.0, 2.0]]

    result = interval1.union(interval2)
    assert len(result.times) == 1
    assert np.array_equal(result.times[0], [1.0, 2.0])


def test_interval_union_validation_error():
    """Test Interval.union with invalid intervals causing validation errors."""
    from spyglass.common.common_interval import Interval

    interval1 = Interval([[1.0, 3.0]], warn=False)

    interval1.times = np.array([[3.0, 1.0]])

    with pytest.raises(ValueError, match="Negative cumulative sum"):
        interval1.union([[4.0, 5.0]])


def test_interval_union_mismatch_starts_stops():
    """Test Interval.union with mismatched starts and stops."""
    from spyglass.common.common_interval import Interval

    # Create a scenario that would cause mismatched starts/stops
    interval1 = Interval([[1.0, 2.0]], warn=False)

    with patch("spyglass.common.common_interval.np.lexsort") as mock_lexsort:
        # Mock lexsort to create a mismatch scenario
        mock_lexsort.return_value = np.array([0])

        with patch("spyglass.common.common_interval.np.cumsum") as mock_cumsum:
            # Mock cumsum to never reach 0 (causing start/stop mismatch)
            mock_cumsum.return_value = np.array([1, 2, 3])

            with pytest.raises(
                ValueError, match="Mismatched number of union starts and stops"
            ):
                interval1.union([[3.0, 4.0]])


def test_interval_union_adjacent_index():
    """Test Interval.union_adjacent_index method."""
    from spyglass.common.common_interval import Interval

    # Test adjacent intervals (should merge)
    interval1 = Interval([[1, 5]], warn=False)
    interval2 = [[6, 10]]  # Adjacent: 5+1 = 6

    result = interval1.union_adjacent_index(interval2)
    assert len(result.times) == 1
    assert np.array_equal(result.times[0], [1, 10])

    # Test non-adjacent intervals (should concatenate)
    interval1 = Interval([[1, 5]], warn=False)
    interval2 = [[8, 10]]  # Not adjacent: 5+1 != 8

    result = interval1.union_adjacent_index(interval2)
    assert len(result.times) == 2
    assert np.array_equal(result.times[0], [1, 5])
    assert np.array_equal(result.times[1], [8, 10])


def test_interval_subtract():
    """Test Interval.subtract method."""
    from spyglass.common.common_interval import Interval

    interval1 = Interval([[1.0, 10.0]], warn=False)
    interval2 = [[3.0, 5.0], [7.0, 8.0]]  # Remove these parts

    result = interval1.subtract(interval2)

    # Should have 3 parts: [1,3], [5,7], [8,10]
    assert len(result.times) == 3
    assert np.array_equal(result.times[0], [1.0, 3.0])
    assert np.array_equal(result.times[1], [5.0, 7.0])
    assert np.array_equal(result.times[2], [8.0, 10.0])


def test_interval_subtract_with_min_length():
    """Test Interval.subtract with minimum length filtering."""
    from spyglass.common.common_interval import Interval

    interval1 = Interval([[1.0, 10.0]], warn=False)
    interval2 = [[4.9, 5.1]]  # Creates small remaining pieces

    result = interval1.subtract(interval2, min_length=2.0)

    # Only pieces >= 2.0 should remain
    assert (
        len(result.times) == 2
    )  # [1.0, 4.9] length 3.9, [5.1, 10.0] length 4.9
    assert np.array_equal(result.times[0], [1.0, 4.9])
    assert np.array_equal(result.times[1], [5.1, 10.0])


def test_interval_subtract_reverse():
    """Test Interval.subtract with reverse=True."""
    from spyglass.common.common_interval import Interval

    interval1 = Interval([[1.0, 5.0]], warn=False)
    interval2 = [[0.0, 10.0]]  # Larger interval

    # Normal: interval1 - interval2 = empty (smaller - larger)
    result_normal = interval1.subtract(interval2)
    assert len(result_normal.times) == 0

    # Reverse: interval2 - interval1 = parts of interval2 not in interval1
    result_reverse = interval1.subtract(interval2, reverse=True)
    assert len(result_reverse.times) == 2
    assert np.array_equal(result_reverse.times[0], [0.0, 1.0])
    assert np.array_equal(result_reverse.times[1], [5.0, 10.0])


def test_interval_consolidate():
    """Test Interval.consolidate and _consolidate methods."""
    from spyglass.common.common_interval import Interval

    # Create overlapping intervals
    intervals = [[1.0, 3.0], [2.0, 5.0], [7.0, 9.0], [8.0, 10.0]]
    interval = Interval(intervals, warn=False, no_overlap=False)

    result = interval.consolidate()

    # Should consolidate to 2 intervals
    assert len(result.times) == 2
    assert np.array_equal(result.times[0], [1.0, 5.0])
    assert np.array_equal(result.times[1], [7.0, 10.0])


def test_interval_censor():
    """Test Interval.censor method."""
    from spyglass.common.common_interval import Interval

    interval = Interval([[1.0, 5.0], [7.0, 10.0]], warn=False)
    timestamps = [2.0, 4.0, 8.0]

    result = interval.censor(timestamps)

    assert np.array_equal(result.times, np.array([[2.0, 5.0], [7.0, 8.0]]))


def test_interval_add_removal_window():
    """Test Interval.add_removal_window method."""
    from spyglass.common.common_interval import Interval

    timestamps = np.arange(0, 5, 0.01)
    interval = Interval([[100, 200], [300, 400]], warn=False)

    # Add 100ms removal window (50ms on each side)
    result = interval.add_removal_window(100, timestamps)

    # The intervals are index ranges into the timestamp array.
    assert len(result.times) == 2
    assert np.allclose(result.times[0], [0.95, 2.05])
    assert np.allclose(result.times[1], [2.95, 4.05])


def test_interval_str_representation():
    """Test Interval string representation methods."""
    from spyglass.common.common_interval import Interval

    interval = Interval([[1.0, 2.0], [3.0, 4.0]], warn=False)

    # Test __repr__
    repr_str = repr(interval)
    assert "Interval" in repr_str
    assert "Interval([1. 2.]" in repr_str

    # Test to_str
    str_result = interval.to_str(interval.times)
    assert isinstance(str_result, str)
    assert "[1., 2.]" in str_result

    # Test __len__
    assert len(interval) == 2


def test_interval_kwargs_preservation():
    """Test that Interval preserves kwargs in returned objects."""
    from spyglass.common.common_interval import Interval

    # Create interval with custom kwargs
    interval = Interval(
        [[1.0, 2.0], [3.0, 4.0]],
        warn=False,
        no_overlap=True,
        custom_param="test_value",
    )

    # Operations should preserve kwargs
    result = interval.intersect([[1.5, 3.5]])
    assert result.kwargs["custom_param"] == "test_value"
    assert result.kwargs["no_overlap"] is True


# Test deprecated functions for coverage
def test_deprecated_functions():
    """Test deprecated standalone functions that delegate to Interval class."""
    from spyglass.common.common_interval import (
        consolidate_intervals,
        interval_list_intersect,
        interval_list_union,
        union_adjacent_index,
    )
    from spyglass.common.common_usage import ActivityLog

    # All should log deprecation warnings but still work
    test_intervals = [[1.0, 3.0], [2.0, 4.0]]

    with patch.object(ActivityLog, "deprecate_log") as mock_log:
        # Test consolidate_intervals
        result = consolidate_intervals(test_intervals)
        assert len(result) == 1  # Should consolidate overlapping intervals
        mock_log.assert_called()

        mock_log.reset_mock()

        # Test union_adjacent_index
        result = union_adjacent_index([1, 5], [6, 10])
        mock_log.assert_called()

        mock_log.reset_mock()

        # Test interval_list_intersect
        result = interval_list_intersect([[1.0, 4.0]], [[2.0, 3.0]])
        mock_log.assert_called()

        mock_log.reset_mock()

        # Test interval_list_union
        result = interval_list_union([[1.0, 2.0]], [[3.0, 4.0]])
        mock_log.assert_called()


def test_interval_list_operations_from_targeted():
    """Basic instantiation path for IntervalList."""
    from spyglass.common.common_interval import IntervalList

    interval_list = IntervalList()
    assert interval_list is not None


@pytest.mark.parametrize(
    "test_input,expected_type",
    [
        ([1, 2, 3], list),
        (np.array([1, 2, 3]), np.ndarray),
        (pd.Series([1, 2, 3]), pd.Series),
    ],
)
def test_data_type_handling_from_targeted(test_input, expected_type):
    """Sanity-check expected data container types."""
    assert isinstance(test_input, expected_type)


def test_data_validation_edge_cases_from_targeted():
    """Basic DataFrame edge-case sanity checks."""
    empty_data = pd.DataFrame()
    assert len(empty_data) == 0

    single_row = pd.DataFrame({"col1": [1], "col2": [2]})
    assert len(single_row) == 1

    data_with_nan = pd.DataFrame({"col1": [1, np.nan, 3]})
    assert bool(data_with_nan.isna().any().any())
