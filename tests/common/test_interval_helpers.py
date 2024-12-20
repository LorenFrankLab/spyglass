import numpy as np
import pytest


@pytest.fixture(scope="session")
def list_intersect(common):
    yield common.common_interval.interval_list_intersect


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
def test_list_intersect(list_intersect, one, two, result):
    assert np.array_equal(
        list_intersect(one, two), result
    ), "Problem with common_interval.interval_list_intersect"


@pytest.fixture(scope="session")
def set_difference(common):
    yield common.common_interval.interval_set_difference_inds


@pytest.mark.parametrize(
    "one, two, expected_result",
    [
        (  # No overlap
            [(0, 5), (8, 10)],
            [(5, 8)],
            [(0, 5), (8, 10)],
        ),
        (  # Overlap
            [(0, 5), (8, 10)],
            [(1, 2), (3, 4), (6, 9)],
            [(0, 1), (2, 3), (4, 5), (9, 10)],
        ),
        (  # One empty
            [],
            [(1, 2), (3, 4), (6, 9)],
            [],
        ),
        (  # Two empty
            [(0, 5), (8, 10)],
            [],
            [(0, 5), (8, 10)],
        ),
        (  # Equal intervals
            [(0, 5), (8, 10)],
            [(0, 5), (8, 10)],
            [],
        ),
        (  # Multiple overlaps
            [(0, 10)],
            [(1, 3), (4, 6), (7, 9)],
            [(0, 1), (3, 4), (6, 7), (9, 10)],
        ),
    ],
)
def test_set_difference(set_difference, one, two, expected_result):
    assert (
        set_difference(one, two) == expected_result
    ), "Problem with common_interval.interval_set_difference_inds"


@pytest.mark.parametrize(
    "expected_result, min_len, max_len",
    [
        (np.array([[0, 1]]), 0.0, 10),
        (np.array([[0, 1], [0, 1e11]]), 0.0, 1e12),
        (np.array([[0, 0], [0, 1]]), -1, 10),
    ],
)
def test_intervals_by_length(common, expected_result, min_len, max_len):
    # input is the same across all tests. Could be parametrized as above
    inds = common.common_interval.intervals_by_length(
        interval_list=np.array([[0, 0], [0, 1], [0, 1e11]]),
        min_length=min_len,
        max_length=max_len,
    )
    assert np.array_equal(
        inds, expected_result
    ), "Problem with common_interval.intervals_by_length"


@pytest.fixture
def interval_list_dict():
    yield {
        "interval_list": np.array([[1, 4], [6, 8]]),
        "timestamps": np.array([0, 1, 5, 7, 8, 9]),
    }


def test_interval_list_contains_ind(common, interval_list_dict):
    idxs = common.common_interval.interval_list_contains_ind(
        **interval_list_dict
    )
    assert np.array_equal(
        idxs, np.array([1, 3, 4])
    ), "Problem with common_interval.interval_list_contains_ind"


def test_interval_list_contains(common, interval_list_dict):
    idxs = common.common_interval.interval_list_contains(**interval_list_dict)
    assert np.array_equal(
        idxs, np.array([1, 7, 8])
    ), "Problem with common_interval.interval_list_contains"


def test_interval_list_excludes_ind(common, interval_list_dict):
    idxs = common.common_interval.interval_list_excludes_ind(
        **interval_list_dict
    )
    assert np.array_equal(
        idxs, np.array([0, 2, 5])
    ), "Problem with common_interval.interval_list_excludes_ind"


def test_interval_list_excludes(common, interval_list_dict):
    idxs = common.common_interval.interval_list_excludes(**interval_list_dict)
    assert np.array_equal(
        idxs, np.array([0, 5, 9])
    ), "Problem with common_interval.interval_list_excludes"


def test_consolidate_intervals_1dim(common):
    exp = common.common_interval.consolidate_intervals(np.array([0, 1]))
    assert np.array_equal(
        exp, np.array([[0, 1]])
    ), "Problem with common_interval.consolidate_intervals"


@pytest.mark.parametrize(
    "interval1, interval2, exp_result",
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
def test_union_adjacent_index(common, interval1, interval2, exp_result):
    assert np.array_equal(
        common.common_interval.union_adjacent_index(interval1, interval2),
        exp_result,
    ), "Problem with common_interval.union_adjacent_index"


@pytest.mark.parametrize(
    "interval1, interval2, exp_result",
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
def test_interval_list_union(common, interval1, interval2, exp_result):
    assert np.array_equal(
        common.common_interval.interval_list_union(interval1, interval2),
        exp_result,
    ), "Problem with common_interval.interval_list_union"


def test_interval_list_censor_error(common):
    with pytest.raises(ValueError):
        common.common_interval.interval_list_censor(
            np.array([[0, 1]]), np.array([2])
        )


def test_interval_list_censor(common):
    assert np.array_equal(
        common.common_interval.interval_list_censor(
            np.array([[0, 2], [4, 5]]), np.array([1, 2, 4])
        ),
        np.array([[1, 2]]),
    ), "Problem with common_interval.interval_list_censor"


@pytest.mark.parametrize(
    "interval_list, exp_result",
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
def test_interval_from_inds(common, interval_list, exp_result):
    assert np.array_equal(
        common.common_interval.interval_from_inds(interval_list),
        exp_result,
    ), "Problem with common_interval.interval_from_inds"


@pytest.mark.parametrize(
    "intervals1, intervals2, min_length, exp_result",
    [
        (
            np.array([[0, 2], [4, 5]]),
            np.array([[1, 3], [2, 4]]),
            0,
            np.array([[0, 1], [4, 5]]),
        ),
        (
            np.array([[0, 2], [4, 5]]),
            np.array([[1, 3], [2, 4]]),
            1,
            np.zeros((0, 2)),
        ),
        (
            np.array([[0, 2], [4, 6]]),
            np.array([[5, 8], [2, 4]]),
            1,
            np.array([[0, 2]]),
        ),
    ],
)
def test_interval_list_complement(
    common, intervals1, intervals2, min_length, exp_result
):
    ic = common.common_interval.interval_list_complement
    assert np.array_equal(
        ic(intervals1, intervals2, min_length),
        exp_result,
    ), "Problem with common_interval.interval_list_compliment"
