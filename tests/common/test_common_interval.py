import numpy as np
from spyglass.common.common_interval import (
    interval_list_intersect,
    interval_set_difference_inds,
)


def test_interval_list_intersect1():
    interval_list1 = np.array([[0, 10], [3, 5], [14, 16]])
    interval_list2 = np.array([[10, 11], [9, 14], [13, 18]])
    intersection_list = interval_list_intersect(interval_list1, interval_list2)
    assert np.all(intersection_list == np.array([[9, 10], [14, 16]]))


def test_interval_list_intersect2():
    # if there is no intersection, return empty list
    interval_list1 = np.array([[0, 10], [3, 5]])
    interval_list2 = np.array([[11, 14]])
    intersection_list = interval_list_intersect(interval_list1, interval_list2)
    assert len(intersection_list) == 0


def test_interval_set_difference_inds_no_overlap():
    intervals1 = [(0, 5), (8, 10)]
    intervals2 = [(5, 8)]
    result = interval_set_difference_inds(intervals1, intervals2)
    assert result == [(0, 5), (8, 10)]


def test_interval_set_difference_inds_overlap():
    intervals1 = [(0, 5), (8, 10)]
    intervals2 = [(1, 2), (3, 4), (6, 9)]
    result = interval_set_difference_inds(intervals1, intervals2)
    assert result == [(0, 1), (2, 3), (4, 5), (9, 10)]


def test_interval_set_difference_inds_empty_intervals1():
    intervals1 = []
    intervals2 = [(1, 2), (3, 4), (6, 9)]
    result = interval_set_difference_inds(intervals1, intervals2)
    assert result == []


def test_interval_set_difference_inds_empty_intervals2():
    intervals1 = [(0, 5), (8, 10)]
    intervals2 = []
    result = interval_set_difference_inds(intervals1, intervals2)
    assert result == [(0, 5), (8, 10)]


def test_interval_set_difference_inds_equal_intervals():
    intervals1 = [(0, 5), (8, 10)]
    intervals2 = [(0, 5), (8, 10)]
    result = interval_set_difference_inds(intervals1, intervals2)
    assert result == []


def test_interval_set_difference_inds_multiple_overlaps():
    intervals1 = [(0, 10)]
    intervals2 = [(1, 3), (4, 6), (7, 9)]
    result = interval_set_difference_inds(intervals1, intervals2)
    assert result == [(0, 1), (3, 4), (6, 7), (9, 10)]
