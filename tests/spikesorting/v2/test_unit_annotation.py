"""Pure contracts for typed unit-annotation content identity."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spyglass.spikesorting.v2._unit_annotation import (
    annotation_set_hash,
    normalize_annotation_value,
    normalize_producer_parameters,
    producer_parameters_hash,
)


def _hash(values, *, value_type="float", parameters=None):
    return annotation_set_hash(
        annotation_name="custom_score",
        annotation_version=1,
        value_type=value_type,
        producer_parameters=parameters or {},
        values=values,
    )


def test_annotation_hash_is_canonical_and_numpy_native():
    """Ordering and NumPy scalar wrappers do not change logical identity."""
    first = _hash(
        [(2, np.float64(0.25)), (1, np.float64(0.5))],
        parameters={"window": np.int64(4), "nested": (True, None)},
    )
    second = _hash(
        [(1, 0.5), (2, 0.25)],
        parameters={"nested": [True, None], "window": 4},
    )
    assert first == second


def test_annotation_hash_distinguishes_special_values_and_precision():
    """None, NaN, empty text, and nearby float64 values stay distinct."""
    assert _hash([(1, None)]) != _hash([(1, math.nan)])
    assert _hash([(1, 1.0000000000000002)]) != _hash([(1, 1.0)])
    assert _hash([(1, "")], value_type="text") != _hash(
        [(1, None)], value_type="text"
    )


def test_annotation_value_types_are_not_coerced():
    """Runtime values must agree with their declared scalar type."""
    assert normalize_annotation_value("int", np.int64(2)) == 2
    assert normalize_annotation_value("bool", np.bool_(True)) is True
    assert normalize_annotation_value("text", "") == ""
    with pytest.raises(TypeError, match="float annotations"):
        normalize_annotation_value("float", 2)
    with pytest.raises(TypeError, match="int annotations"):
        normalize_annotation_value("int", True)
    with pytest.raises(TypeError, match="text annotations"):
        normalize_annotation_value("text", 2.0)


def test_producer_parameters_are_normalized_and_hashed():
    """The stored normalized mapping and its hash share one canonical form."""
    supplied = {"threshold": np.float64(0.25), "flags": (True, None)}
    normalized = normalize_producer_parameters(supplied)
    assert normalized == {
        "flags": [True, None],
        "threshold": {"__float__": "0x1.0000000000000p-2"},
    }
    assert producer_parameters_hash(supplied) == producer_parameters_hash(
        normalized
    )
