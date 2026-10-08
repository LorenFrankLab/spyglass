"""Native numerical contracts shared by scientific computation boundaries.

Validation preserves input order and repeated events. It never rounds frame or
unit identifiers, guesses array dimensions, or repairs nonfinite observations.
Only the standard library and NumPy are required; no table or I/O dependencies.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

_INT64 = np.iinfo(np.int64)


def integer_scalar(value, *, name: str, nonnegative: bool = False) -> int:
    """Require a native integer representable by the storage int64 dtype."""
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer; got {value!r}.")
    result = int(value)
    if result < _INT64.min or result > _INT64.max:
        raise ValueError(f"{name} must fit in int64; got {value!r}.")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative; got {value!r}.")
    return result


def _array(values, *, name):
    try:
        return np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numerical array.") from exc


def integer_vector(values, *, name: str, nonnegative: bool = False):
    """Return a 1D int64 vector without lossy conversion or reordering.

    Python sequences are checked before NumPy can merge booleans with integers
    or cast large unsigned values. Native integer ndarrays use vector checks.
    Empty 1D inputs remain valid regardless of their inferred dtype.
    """
    array = _array(values, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    if not array.size:
        return np.empty(0, dtype=np.int64)
    if not isinstance(values, np.ndarray) or array.dtype.kind == "O":
        result = np.asarray(
            [
                integer_scalar(value, name=name, nonnegative=nonnegative)
                for value in values
            ],
            dtype=np.int64,
        )
    else:
        if array.dtype.kind not in "iu":
            raise ValueError(
                f"{name} must contain integers, without booleans or rounding."
            )
        if array.dtype.kind == "u" and np.any(array > np.uint64(_INT64.max)):
            raise ValueError(f"{name} must fit in int64.")
        result = array.astype(np.int64, copy=False)
        if nonnegative and np.any(result < 0):
            raise ValueError(f"{name} must contain nonnegative integers.")
    return result


def finite_scalar(
    value, *, name: str, positive: bool = False, nonnegative: bool = False
) -> float:
    """Require a finite real scalar, with optional sign constraints."""
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive.")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _finite_array(array, *, name, copy, readonly):
    if array.size and array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain finite real numbers.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = (
            array.astype(np.float64, copy=False)
            if array.size
            else np.empty(array.shape, dtype=np.float64)
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    if copy or readonly:
        result = result.copy()
    if readonly:
        result.flags.writeable = False
    return result


def finite_vector(
    values, *, name: str, copy: bool = False, readonly: bool = False
):
    """Return a finite float64 1D vector, preserving order and duplicates.

    ``readonly=True`` returns an owned copy so the caller's array stays writable.
    """
    array = _array(values, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return _finite_array(array, name=name, copy=copy, readonly=readonly)


def finite_intervals(
    values,
    *,
    name: str,
    allow_zero_length: bool = True,
    copy: bool = False,
    readonly: bool = False,
):
    """Return finite float64 ``(n, 2)`` intervals without sorting or merging.

    Empty ``[]`` denotes ``(0, 2)``. Reversed intervals are always rejected;
    zero-length intervals are allowed by default. ``readonly=True`` owns a copy.
    """
    array = _array(values, name=name)
    if array.ndim == 1 and not array.size:
        array = array.reshape(0, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")
    result = _finite_array(array, name=name, copy=copy, readonly=readonly)
    invalid = result[:, 1] < result[:, 0]
    if not allow_zero_length:
        invalid |= result[:, 1] == result[:, 0]
    if np.any(invalid):
        relation = "greater than" if not allow_zero_length else "at least"
        raise ValueError(f"{name} stop must be {relation} its start.")
    return result
