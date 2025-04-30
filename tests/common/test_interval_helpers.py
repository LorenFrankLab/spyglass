import numpy as np
import pytest


@pytest.fixture(scope="session")
def interval_obj(common):
    yield common.common_interval.Interval


@pytest.mark.parametrize(
    "one, two, result",
    [
        (
            np.array([[0, 10], [3, 5], [14, 16]]),
            np.array([[10, 11], [9, 14], [13, 18]]),
            np.array([[9, 10], [14, 16]]),
        ),
        (  # Empty result for no intersection
            np.array([[0, 10], [3, 5]]),
            np.array([[11, 14]]),
            np.array([]),
        ),
    ],
)
def test_list_intersect(interval_obj, one, two, result):
    assert np.array_equal(
        interval_obj(one).intersect(two).times, result
    ), "Problem with common_interval.Interval.intersect"


@pytest.mark.parametrize(
    "one, two, min_length, expected_result",
    [
        (  # Indices, No overlap
            [(0, 5), (8, 10)],
            [(5, 8)],
            None,
            [(0, 5), (8, 10)],
        ),
        (  # Indices, Overlap
            [(0, 5), (8, 10)],
            [(1, 2), (3, 4), (6, 9)],
            None,
            [(0, 1), (2, 3), (4, 5), (9, 10)],
        ),
        (  # Indices, One empty
            [],
            [(1, 2), (3, 4), (6, 9)],
            None,
            [],
        ),
        (  # Indices, Two empty
            [(0, 5), (8, 10)],
            [],
            None,
            [(0, 5), (8, 10)],
        ),
        (  # Indices, Equal intervals
            [(0, 5), (8, 10)],
            [(0, 5), (8, 10)],
            None,
            [],
        ),
        (  # Indices, Multiple overlaps
            [(0, 10)],
            [(1, 3), (4, 6), (7, 9)],
            None,
            [(0, 1), (3, 4), (6, 7), (9, 10)],
        ),
        (  # Arrays, Overlap
            np.array([[0, 2], [4, 5]]),
            np.array([[1, 3], [2, 4]]),
            0,
            np.array([[0, 1], [4, 5]]),
        ),
        (  # Arrays, Min length
            np.array([[0, 2], [4, 5]]),
            np.array([[1, 3], [2, 4]]),
            1,
            np.zeros((0, 2)),
        ),
        (  # Arrays, No overlap
            np.array([[0, 2], [4, 6]]),
            np.array([[5, 8], [2, 4]]),
            1,
            np.array([[0, 2]]),
        ),
    ],
)
def test_subtract(interval_obj, one, two, min_length, expected_result):
    ret = interval_obj(one).subtract(two, min_length=min_length)
    assert np.array_equal(
        ret.times, expected_result
    ), "Problem with Interval.subtract"


@pytest.mark.parametrize(
    "expected_result, min_len, max_len",
    [
        (np.array([[0, 1]]), 0.0, 10),
        (np.array([[0, 1], [0, 1e11]]), 0.0, 1e12),
        (np.array([[0, 0], [0, 1]]), -1, 10),
    ],
)
def test_intervals_by_length(interval_obj, expected_result, min_len, max_len):
    inds = interval_obj(np.array([[0, 0], [0, 1], [0, 1e11]])).by_length(
        min_length=min_len, max_length=max_len
    )
    assert np.array_equal(
        inds.times, expected_result
    ), "Problem with Interval.by_length"


@pytest.fixture
def example_interval() -> tuple:
    yield np.array([[1, 4], [6, 8]]), np.array([0, 1, 5, 7, 8, 9])


def test_interval_list_contains_ind(interval_obj, example_interval):
    interval, timestamps = example_interval
    idxs = interval_obj(interval).contains(timestamps, as_indices=True)
    assert np.array_equal(
        idxs.times, np.array([1, 3, 4])
    ), "Problem with Interval.contains"


def test_interval_list_contains(interval_obj, example_interval):
    interval, timestamps = example_interval
    idxs = interval_obj(interval).contains(timestamps)
    assert np.array_equal(
        idxs.times, np.array([1, 7, 8])
    ), "Problem with Interval.contains"


def test_interval_list_excludes_ind(interval_obj, example_interval):
    interval, timestamps = example_interval
    idxs = interval_obj(interval).excludes(timestamps, as_indices=True)
    assert np.array_equal(
        idxs.times, np.array([0, 2, 5])
    ), "Problem with Interval.excludes"


def test_interval_list_excludes(interval_obj, example_interval):
    interval, timestamps = example_interval
    idxs = interval_obj(interval).excludes(timestamps)
    assert np.array_equal(
        idxs.times, np.array([0, 5, 9])
    ), "Problem with Interval.excludes"


def test_consolidate_intervals_1dim(interval_obj):
    exp = interval_obj(np.array([0, 1])).consolidate().times
    assert np.array_equal(
        exp, np.array([[0, 1]])
    ), "Problem with Interval.consolidate"


@pytest.mark.parametrize(
    "one, two, expected_result",
    [
        (
            np.array([[0, 1]]),
            np.array([[2, 3]]),
            np.array([[0, 3]]),
        ),
        (
            np.array([[2, 3]]),
            np.array([[0, 1]]),
            np.array([[0, 3]]),
        ),
        (
            np.array([[0, 3]]),
            np.array([[2, 4]]),
            np.array([[0, 3], [2, 4]]),
        ),
    ],
)
def test_union_adjacent_index(interval_obj, one, two, expected_result):
    ret = interval_obj(one).union_adjacent_index(two).times
    assert np.array_equal(
        ret, expected_result
    ), "Problem with Interval.union_adjacent_index"


@pytest.mark.parametrize(
    "one, two, expected_result",
    [
        (
            np.array([[0, 3]]),
            np.array([[2, 4]]),
            np.array([[0, 4]]),
        ),
        (
            np.array([[0, -1]]),
            np.array([[2, 4]]),
            np.array([[2, 0]]),
        ),
        (
            np.array([[0, 1]]),
            np.array([[2, 1e11]]),
            np.array([[0, 1], [2, 1e11]]),
        ),
    ],
)
def test_interval_list_union(interval_obj, one, two, expected_result):
    ret = interval_obj(one).union(two).times
    assert np.array_equal(ret, expected_result), "Problem with Interval.union"


def test_interval_list_censor_error(interval_obj):
    with pytest.raises(ValueError):
        interval_obj(np.array([[0, 1]])).censor(np.array([2]))


def test_interval_list_censor(interval_obj):
    ret = interval_obj(np.array([[0, 2], [4, 5]])).censor(np.array([1, 2, 4]))
    assert np.array_equal(
        ret.times, np.array([[1, 2]])
    ), "Problem with Interval.censor"


@pytest.mark.parametrize(
    "interval_list, expected_result",
    [
        (
            np.array([0, 1, 2, 3, 6, 7, 8, 9]),
            np.array([[0, 3], [6, 9]]),
        ),
        (
            np.array([0, 1, 2]),
            np.array([[0, 2]]),
        ),
        (
            np.array([2, 3, 1, 0]),
            np.array([[0, 3]]),
        ),
        (
            np.array([2, 3, 0]),
            np.array([[0, 0], [2, 3]]),
        ),
    ],
)
def test_interval_from_inds(interval_obj, interval_list, expected_result):
    assert np.array_equal(
        interval_obj(interval_list, from_inds=True).times, expected_result
    ), "Problem with Interval(x, from_inds=True)"
