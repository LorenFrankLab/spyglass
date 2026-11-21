"""Tests for the numerical argument validators in spyglass.utils.numerical."""

from fractions import Fraction

import numpy as np
import pytest

from spyglass.utils.numerical import (
    finite_intervals,
    finite_scalar,
    finite_vector,
    integer_scalar,
    integer_vector,
)

pytestmark = pytest.mark.unit

INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)
RAGGED = [[1], [1, 2]]


# integer_scalar and integer_vector


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),
        (-5, -5),
        (np.int32(7), 7),
        (np.uint64(INT64_MAX), INT64_MAX),
        (INT64_MAX, INT64_MAX),
        (INT64_MIN, INT64_MIN),
    ],
)
def test_integer_scalar_returns_python_int(value, expected):
    got = integer_scalar(value, name="unit_id")
    assert type(got) is int
    assert got == expected


@pytest.mark.parametrize(
    "value, match",
    [
        pytest.param(True, "must be an integer", id="bool"),
        pytest.param(np.bool_(False), "must be an integer", id="np-bool"),
        pytest.param(1.0, "must be an integer", id="integral-float"),
        pytest.param(np.float64(2.0), "must be an integer", id="np-float"),
        pytest.param("1", "must be an integer", id="str"),
        pytest.param(None, "must be an integer", id="none"),
        pytest.param([1], "must be an integer", id="list"),
        pytest.param(INT64_MAX + 1, "must fit in int64", id="above-int64"),
        pytest.param(INT64_MIN - 1, "must fit in int64", id="below-int64"),
        pytest.param(np.uint64(2**63), "must fit in int64", id="big-uint64"),
    ],
)
def test_integer_scalar_rejects_non_integers_and_overflow(value, match):
    with pytest.raises(ValueError, match=f"unit_id {match}"):
        integer_scalar(value, name="unit_id")


def test_integer_scalar_nonnegative():
    assert integer_scalar(0, name="unit_id", nonnegative=True) == 0
    with pytest.raises(ValueError, match="unit_id must be nonnegative"):
        integer_scalar(-1, name="unit_id", nonnegative=True)


@pytest.mark.parametrize(
    "values",
    [
        [3, 1, 3],
        (3, 1, 3),
        [np.int64(3), 1, np.uint16(3)],
        np.array([3, 1, 3], dtype=np.int32),
        np.array([3, 1, 3], dtype=np.uint8),
        np.array([3, 1, 3], dtype=object),
    ],
)
def test_integer_vector_returns_int64_in_input_order(values):
    got = integer_vector(values, name="frames")
    assert got.dtype == np.int64
    assert got.tolist() == [3, 1, 3]


def test_integer_vector_keeps_int64_limits():
    uint = np.array([INT64_MAX, 0], dtype=np.uint64)
    assert integer_vector(uint, name="frames").tolist() == [INT64_MAX, 0]
    assert uint.dtype == np.uint64
    got = integer_vector([INT64_MIN, INT64_MAX], name="frames")
    assert got.tolist() == [INT64_MIN, INT64_MAX]


@pytest.mark.parametrize(
    "values", [[], (), np.array([]), np.array([], dtype=np.uint8)]
)
def test_integer_vector_accepts_empty_1d(values):
    got = integer_vector(values, name="frames")
    assert got.dtype == np.int64
    assert got.shape == (0,)


@pytest.mark.parametrize(
    "values, match",
    [
        pytest.param([1, True], "must be an integer", id="list-bool"),
        pytest.param([False], "must be an integer", id="list-only-bool"),
        pytest.param([1.0], "must be an integer", id="list-integral-float"),
        pytest.param([1.9], "must be an integer", id="list-float"),
        pytest.param(["1"], "must be an integer", id="list-str"),
        pytest.param([None], "must be an integer", id="list-none"),
        pytest.param(
            np.array([1, True], dtype=object),
            "must be an integer",
            id="object-array-bool",
        ),
        pytest.param([INT64_MAX + 1], "must fit in int64", id="list-above"),
        pytest.param([INT64_MIN - 1], "must fit in int64", id="list-below"),
        pytest.param(
            np.array([2**63], dtype=np.uint64),
            "must fit in int64",
            id="big-uint64-array",
        ),
        pytest.param(np.array([True]), "must contain integers", id="bool-arr"),
        pytest.param(np.array([1.0]), "must contain integers", id="float-arr"),
        pytest.param(np.array(["1"]), "must contain integers", id="str-arr"),
        pytest.param(5, "must be a one-dimensional", id="scalar"),
        pytest.param([[1, 2]], "must be a one-dimensional", id="2d"),
        pytest.param(
            np.empty((0, 2), dtype=np.int64),
            "must be a one-dimensional",
            id="empty-2d",
        ),
        pytest.param(RAGGED, "must be a numerical array", id="ragged"),
    ],
)
def test_integer_vector_rejects_lossy_values_and_wrong_shapes(values, match):
    with pytest.raises(ValueError, match=f"frames {match}"):
        integer_vector(values, name="frames")


@pytest.mark.parametrize("values", [[0, 2], np.array([0, 2])])
def test_integer_vector_nonnegative_accepts_zero(values):
    got = integer_vector(values, name="frames", nonnegative=True)
    assert got.tolist() == [0, 2]


@pytest.mark.parametrize("values", [[0, -1], np.array([0, -1])])
def test_integer_vector_nonnegative_rejects_negative(values):
    with pytest.raises(ValueError, match="frames must .*nonnegative"):
        integer_vector(values, name="frames", nonnegative=True)


# finite_scalar and finite_vector


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0.0),
        (-2.5, -2.5),
        (np.float32(0.5), 0.5),
        (np.int64(3), 3.0),
        (Fraction(1, 4), 0.25),
    ],
)
def test_finite_scalar_returns_python_float(value, expected):
    got = finite_scalar(value, name="duration")
    assert type(got) is float
    assert got == expected


@pytest.mark.parametrize(
    "value, match",
    [
        pytest.param(True, "must be a finite real scalar", id="bool"),
        pytest.param(np.bool_(True), "must be a finite real", id="np-bool"),
        pytest.param("1", "must be a finite real scalar", id="str"),
        pytest.param(None, "must be a finite real scalar", id="none"),
        pytest.param([1.0], "must be a finite real scalar", id="list"),
        pytest.param(np.array([1.0]), "must be a finite real", id="array"),
        pytest.param(1j, "must be a finite real scalar", id="complex"),
        pytest.param(10**400, "must be a finite real", id="float-overflow"),
        pytest.param(np.nan, "must be finite", id="nan"),
        pytest.param(np.inf, "must be finite", id="inf"),
        pytest.param(-np.inf, "must be finite", id="-inf"),
    ],
)
def test_finite_scalar_rejects_non_real_and_nonfinite(value, match):
    with pytest.raises(ValueError, match=f"duration {match}"):
        finite_scalar(value, name="duration")


def test_finite_scalar_sign_constraints():
    assert finite_scalar(1e-12, name="duration", positive=True) == 1e-12
    assert finite_scalar(0, name="origin", nonnegative=True) == 0.0
    with pytest.raises(ValueError, match="duration must be positive"):
        finite_scalar(0, name="duration", positive=True)
    with pytest.raises(ValueError, match="duration must be positive"):
        finite_scalar(-1, name="duration", positive=True)
    with pytest.raises(ValueError, match="origin must be nonnegative"):
        finite_scalar(-1e-12, name="origin", nonnegative=True)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, -2.0, 1.0],
        np.array([1, -2, 1]),
        np.array([1.0, -2.0, 1.0], dtype=np.float32),
    ],
)
def test_finite_vector_returns_float64_in_input_order(values):
    got = finite_vector(values, name="spike_times")
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, [1.0, -2.0, 1.0])


@pytest.mark.parametrize("values", [[], np.array([], dtype=np.int64)])
def test_finite_vector_accepts_empty_1d(values):
    got = finite_vector(values, name="spike_times")
    assert got.dtype == np.float64
    assert got.shape == (0,)


@pytest.mark.parametrize(
    "values, match",
    [
        pytest.param([np.nan], "must contain only finite", id="nan"),
        pytest.param([0.0, np.inf], "must contain only finite", id="inf"),
        pytest.param([-np.inf], "must contain only finite", id="-inf"),
        pytest.param(
            np.array([np.longdouble("1e400")]),
            "must contain only finite",
            id="float64-overflow",
        ),
        pytest.param([True], "must contain finite real", id="bool"),
        pytest.param([1j], "must contain finite real", id="complex"),
        pytest.param(["1"], "must contain finite real", id="str"),
        pytest.param([None], "must contain finite real", id="none"),
        pytest.param(1.0, "must be a one-dimensional", id="scalar"),
        pytest.param([[1.0]], "must be a one-dimensional", id="2d"),
        pytest.param(RAGGED, "must be a numerical array", id="ragged"),
    ],
)
def test_finite_vector_rejects_nonfinite_non_real_and_wrong_shapes(
    values, match
):
    with pytest.raises(ValueError, match=f"spike_times {match}"):
        finite_vector(values, name="spike_times")


# finite_intervals


@pytest.mark.parametrize(
    "values",
    [
        [[2.0, 3.0], [0.0, 2.0], [1.0, 1.0]],
        np.array([[2, 3], [0, 2], [1, 1]]),
    ],
)
def test_finite_intervals_keep_order_overlap_and_zero_length(values):
    got = finite_intervals(values, name="intervals")
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, [[2.0, 3.0], [0.0, 2.0], [1.0, 1.0]])


@pytest.mark.parametrize(
    "values", [[], np.empty((0,)), np.empty((0, 2), dtype=np.int64)]
)
def test_finite_intervals_accept_empty_as_zero_by_two(values):
    got = finite_intervals(values, name="intervals")
    assert got.dtype == np.float64
    assert got.shape == (0, 2)


def test_finite_intervals_readonly_returns_owned_copy():
    values = np.array([[0.0, 1.0], [2.0, 3.0]])
    got = finite_intervals(values, name="intervals", readonly=True)
    np.testing.assert_array_equal(got, values)
    assert not got.flags.writeable
    assert not np.shares_memory(got, values)
    assert values.flags.writeable


@pytest.mark.parametrize(
    "values, match",
    [
        pytest.param([0, 1], "must have shape", id="flat-pair"),
        pytest.param([[0, 1, 2]], "must have shape", id="three-columns"),
        pytest.param(np.empty((0, 3)), "must have shape", id="empty-3-cols"),
        pytest.param(np.zeros((1, 2, 2)), "must have shape", id="3d"),
        pytest.param(1.0, "must have shape", id="scalar"),
        pytest.param([[2, 1]], "stop must be at least its start", id="rev"),
        pytest.param(
            [[0, 1], [3, 2.5]],
            "stop must be at least its start",
            id="second-row-reversed",
        ),
        pytest.param([[0, np.inf]], "must contain only finite", id="inf"),
        pytest.param([[np.nan, 1]], "must contain only finite", id="nan"),
        pytest.param([[False, True]], "must contain finite real", id="bool"),
        pytest.param([[0j, 1]], "must contain finite real", id="complex"),
        pytest.param([["0", "1"]], "must contain finite real", id="str"),
        pytest.param(RAGGED, "must be a numerical array", id="ragged"),
    ],
)
def test_finite_intervals_reject_bad_shapes_order_and_values(values, match):
    with pytest.raises(ValueError, match=f"intervals {match}"):
        finite_intervals(values, name="intervals")
