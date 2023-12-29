import numpy as np
import pytest
from numpy import all, array


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
