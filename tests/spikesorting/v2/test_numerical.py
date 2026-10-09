"""Native scientific numerical contracts reject loss and preserve observations."""

import numpy as np
import pytest

from spyglass.spikesorting.v2._core.numerical import (
    finite_intervals,
    finite_scalar,
    finite_vector,
    integer_scalar,
    integer_vector,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "values",
    [
        [1, True],
        [False],
        [1.9],
        [1.0],
        ["1"],
        [2**63],
        [-(2**63) - 1],
        np.array([2**63], dtype=np.uint64),
        np.array([True]),
        np.array([[1, 2]]),
    ],
)
def test_frame_vector_rejects_lossy_values_and_wrong_dimensions(values):
    with pytest.raises(ValueError, match="frames"):
        integer_vector(values, name="frames")


def test_frame_vector_preserves_order_duplicates_and_int64_limit():
    values = np.array([2**63 - 1, 3, 3, 0], dtype=np.uint64)
    got = integer_vector(values, name="frames", nonnegative=True)
    assert got.dtype == np.int64
    assert got.tolist() == [2**63 - 1, 3, 3, 0]
    assert values.dtype == np.uint64
    assert integer_vector([], name="frames").shape == (0,)
    with pytest.raises(ValueError, match="nonnegative"):
        integer_vector([0, -1], name="frames", nonnegative=True)


@pytest.mark.parametrize("value", [True, 1.0, "1", 2**63, -(2**63) - 1])
def test_unit_scalar_rejects_lossy_or_unrepresentable_identity(value):
    with pytest.raises(ValueError, match="unit_id"):
        integer_scalar(value, name="unit_id")


@pytest.mark.parametrize(
    "values", [[np.nan], [np.inf], [-np.inf], [[1.0]], [1j], ["1"]]
)
def test_spike_vector_requires_finite_real_one_dimensional_data(values):
    with pytest.raises(ValueError, match="spike_times"):
        finite_vector(values, name="spike_times")


def test_spike_vector_preserves_order_and_duplicates():
    values = np.array([1.0, -2.0, 1.0])
    np.testing.assert_array_equal(
        finite_vector(values, name="spike_times"), values
    )
    assert finite_vector([], name="spike_times").shape == (0,)


@pytest.mark.parametrize(
    "intervals",
    [[0, 1], [[0, 1, 2]], [[2, 1]], [[0, np.inf]], np.empty((0, 3))],
)
def test_intervals_require_finite_ordered_endpoint_pairs(intervals):
    with pytest.raises(ValueError, match="intervals"):
        finite_intervals(intervals, name="intervals")


def test_intervals_preserve_zeros_overlap_and_input_order_for_owning_algorithm():
    values = np.array([[2.0, 3.0], [0.0, 2.0], [1.0, 1.0]])
    got = finite_intervals(values, name="intervals", readonly=True)
    np.testing.assert_array_equal(got, values)
    assert not np.shares_memory(got, values)
    assert values.flags.writeable
    assert not got.flags.writeable
    assert finite_intervals([], name="intervals").shape == (0, 2)


@pytest.mark.parametrize("value", [True, np.nan, np.inf, "1", [1]])
def test_scalar_requires_finite_native_value(value):
    with pytest.raises(ValueError, match="duration"):
        finite_scalar(value, name="duration")


def test_positive_duration_and_nonnegative_origin_constraints_are_explicit():
    assert finite_scalar(np.float64(0.2), name="duration", positive=True) == 0.2
    assert finite_scalar(0, name="origin", nonnegative=True) == 0
    with pytest.raises(ValueError, match="positive"):
        finite_scalar(0, name="duration", positive=True)
    with pytest.raises(ValueError, match="nonnegative"):
        finite_scalar(-1, name="origin", nonnegative=True)
