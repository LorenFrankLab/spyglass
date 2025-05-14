import numpy as np
import pytest


def test_fetch_interval_error(interval_list):
    query = interval_list & 'interval_list_name = "FAKE"'
    with pytest.raises(ValueError):
        query.fetch_interval()


@pytest.fixture(scope="session")
def cautious_interval(interval_list, mini_dict):
    insert = dict(
        mini_dict,
        interval_list_name="TEST CAUTION",
        valid_times=[[0, 1]],
        pipeline="",
    )
    interval_list.insert1(insert, skip_duplicates=True)
    yield insert


def test_cautious_insert_update_error(cautious_interval, interval_list):
    insert = dict(cautious_interval, valid_times=[[0, 5]])
    with pytest.raises(ValueError):
        interval_list.cautious_insert(insert, update=False)


def test_cautious_insert_update(cautious_interval, interval_list):
    new_times = [[0, 2]]
    insert = dict(cautious_interval, valid_times=new_times)
    interval_list.cautious_insert(insert, update=True)
    query = interval_list & 'interval_list_name = "TEST CAUTION"'
    assert np.array_equal(
        new_times,
        query.fetch_interval().times,
    ), "Problem with cautious_insert(update=True)"


def test_cautious_insert_match(interval_list):
    key = (interval_list & 'interval_list_name = "TEST CAUTION"').fetch1()
    before = len(interval_list)
    interval_list.cautious_insert(key, update=False)
    after = len(interval_list)
    assert before == after, "Problem with cautious_insert(update=False)"


@pytest.fixture(scope="session")
def interval_obj(common):
    yield common.common_interval.Interval


def test_interval_no_duplicates(interval_obj):
    # Test with a list of intervals
    intervals = np.array([[0, 1], [0, 1], [3, 4]])
    obj = interval_obj(intervals, no_duplicates=True)
    assert len(obj) == 2, "Problem Interval.__init__ no_duplicates"


def test_interval_no_overlap(interval_obj):
    # Test with a list of intervals
    intervals = np.array([[0, 2], [1, 3], [5, 6]])
    times = interval_obj(intervals, no_overlap=True).times
    expected = np.array([[0, 3], [5, 6]])
    assert np.array_equal(
        times, expected
    ), "Problem Interval.__init__ no_overlap"


def test_interval_repr(interval_obj):
    # Test with a list of intervals
    intervals = np.array([[0, 2], [1, 3], [5, 6]])
    obj = interval_obj(intervals)
    expected = "Interval([0 2]\n\t[1 3]\n\t[5 6])"
    assert repr(obj) == expected, "Problem Interval.__repr__"


def test_interval_getitem(interval_obj):
    # Test with a list of intervals
    intervals = np.array([[0, 2], [5, 6]])
    obj = interval_obj(intervals)[1:].times
    assert np.array_equal(obj, intervals[1:]), "Problem Interval.__getitem__"


def test_interval_getitem_err(interval_obj):
    obj = interval_obj(np.array([[0, 2], [5, 6]]))
    with pytest.raises(ValueError):
        obj["invalid_key"]  # Invalid key for __getitem__


def test_interval_iter(interval_obj):
    # Test with a list of intervals
    intervals = np.array([[0, 2], [5, 6]])
    obj = interval_obj(intervals)
    assert np.array_equal(
        [interval for interval in obj], intervals
    ), "Problem Interval.__iter__"


def test_interval_len(interval_obj):
    intervals = np.array([[0, 2], [5, 6]])
    obj = interval_obj(intervals)
    assert len(obj) == len(intervals), "Problem Interval.__len__"


def test_interval_setter(interval_obj):
    obj = interval_obj(np.array([[0, 2], [5, 6]]))
    key = dict(interval_list_name="TEST", pipeline="FAKE")
    obj.set_key(**key)
    ret = {k: v for k, v in obj.as_dict.items() if k != "valid_times"}
    assert ret == key, "Problem Interval.set_key"


def test_interval_pk(interval_obj):
    obj = interval_obj(np.array([[0, 2]]), name="TEST", nwb_file_name="OTHER")
    assert (
        obj.primary_key["interval_list_name"] == "TEST"
    ), "Problem Interval.primary_key"


def test_interval_pk_err(interval_obj):
    obj = interval_obj(np.array([[0, 2], [5, 6]]))
    with pytest.raises(ValueError):
        obj.primary_key  # No name set


def test_interval_equal(interval_obj):
    intervals = np.array([[0, 2], [5, 6]])
    obj1 = interval_obj(intervals)
    obj2 = interval_obj(intervals)
    assert obj1 == obj2, "Problem Interval.__eq__"


def test_interval_from_interval(interval_obj):
    intervals = np.array([[0, 2], [5, 6]])
    obj1 = interval_obj(intervals)
    obj2 = interval_obj(obj1)
    assert obj1 == obj2, "Problem Interval._extract using Interval"


def test_interval_from_key(interval_list, interval_obj):
    key = interval_list.fetch("KEY", as_dict=True, limit=1)[0]
    query = interval_list & key
    obj_from_key = interval_obj(key)
    obj_from_tbl = query.fetch_interval()
    assert obj_from_key == obj_from_tbl, "Problem Interval._import_from_table"


def test_interval_type_err(interval_obj):
    with pytest.raises(TypeError):
        interval_obj("invalid_type")  # Invalid type for Interval


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


def test_interval_to_seconds(interval_obj, example_interval):
    interval, _ = example_interval
    obj = interval_obj(interval)
    seconds = obj.to_seconds(np.arange(10))
    assert np.array_equal(seconds, interval), "Problem with Interval.to_seconds"


def test_interval_list_contains_ind(interval_obj, example_interval):
    interval, timestamps = example_interval
    idxs = interval_obj(interval).contains(timestamps, as_indices=True)
    assert np.array_equal(
        idxs, np.array([1, 3, 4])
    ), "Problem with Interval.contains"


def test_interval_list_contains(interval_obj, example_interval):
    interval, timestamps = example_interval
    idxs = interval_obj(interval).contains(timestamps)
    assert np.array_equal(
        idxs, np.array([1, 7, 8])
    ), "Problem with Interval.contains"


def test_interval_list_contains_padding(interval_obj, example_interval):
    interval, timestamps = example_interval
    idxs = interval_obj(interval).contains(timestamps, padding=1)
    assert np.array_equal(
        idxs, np.array([0, 7, 9])
    ), "Problem with Interval.contains padding"


def test_interval_list_excludes_ind(interval_obj, example_interval):
    interval, timestamps = example_interval
    idxs = interval_obj(interval).excludes(timestamps, as_indices=True)
    assert np.array_equal(
        idxs, np.array([0, 2, 5])
    ), "Problem with Interval.excludes"


def test_interval_list_excludes(interval_obj, example_interval):
    interval, timestamps = example_interval
    idxs = interval_obj(interval).excludes(timestamps)
    assert np.array_equal(
        idxs, np.array([0, 5, 9])
    ), "Problem with Interval.excludes"


def test_union_consolidate(interval_obj):
    interval = np.array([[0, 3], [2, 3], [5, 8], [7, 9]])
    obj = interval_obj(interval).union_consolidate()
    assert np.array_equal(
        obj.times, np.array([[0, 3], [5, 9]])
    ), "Problem with Interval.union_consolidate"


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


def test_union_adjacent_consolidate(interval_obj):
    one = np.array([[0, 3], [2, 5]])
    ret = interval_obj(one).union_adjacent_consolidate().times
    assert np.array_equal(
        ret, one
    ), "Problem Interval.union_adjacent_consolidate"


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
