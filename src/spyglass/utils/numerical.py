"""Validators for numerical function arguments.

Each validator returns a checked int, float, or NumPy array, or raises
``ValueError``. None of them rounds, reorders, or repairs nonfinite values.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike

_INT64 = np.iinfo(np.int64)


def integer_scalar(
    value: object, *, name: str, nonnegative: bool = False
) -> int:
    """Require a native integer representable by the storage int64 dtype.

    Parameters
    ----------
    value : object
        Candidate integer. Python and NumPy integers are accepted; booleans
        and floats, even integral ones such as ``1.0``, are rejected.
    name : str
        Argument name used in the error message.
    nonnegative : bool, optional
        If True, reject negative values. Default False.

    Returns
    -------
    int
        ``value`` as a Python int.

    Raises
    ------
    ValueError
        If ``value`` is not an integer, does not fit in int64, or is negative
        when ``nonnegative`` is True.
    """
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


def integer_vector(
    values: ArrayLike, *, name: str, nonnegative: bool = False
) -> np.ndarray:
    """Return a 1D int64 vector without lossy conversion or reordering.

    Python sequences are checked element-wise so NumPy cannot coerce
    booleans or cast large unsigned values. Empty 1D inputs are valid.

    Parameters
    ----------
    values : array_like, shape (n_values,)
        Integer values. A NumPy array needs an integer dtype; an object
        array or any other sequence is checked element by element with
        `integer_scalar`.
    name : str
        Argument name used in the error message.
    nonnegative : bool, optional
        If True, reject negative values. Default False.

    Returns
    -------
    np.ndarray, shape (n_values,)
        int64 values in input order, duplicates kept.

    Raises
    ------
    ValueError
        If ``values`` is not one-dimensional, holds a non-integer (booleans
        and integral floats included) or a value outside int64, or holds a
        negative value when ``nonnegative`` is True.
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
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    """Require a finite real scalar, with optional sign constraints.

    Parameters
    ----------
    value : object
        Candidate real number. Python and NumPy integers and floats are
        accepted; booleans, strings, complex numbers, and arrays are rejected.
    name : str
        Argument name used in the error message.
    positive : bool, optional
        If True, reject values ``<= 0``. Default False.
    nonnegative : bool, optional
        If True, reject values ``< 0``. Default False.

    Returns
    -------
    float
        ``value`` as a Python float.

    Raises
    ------
    ValueError
        If ``value`` is not a real number, is NaN or infinite, overflows
        float64, or breaks a requested sign constraint.
    """
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


def _finite_array(array, *, name, readonly=False):
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
    if readonly:
        result = result.copy()
        result.flags.writeable = False
    return result


def finite_vector(values: ArrayLike, *, name: str) -> np.ndarray:
    """Return a finite float64 1D vector, preserving order and duplicates.

    Parameters
    ----------
    values : array_like, shape (n_values,)
        Real values with an integer or float dtype.
    name : str
        Argument name used in the error message.

    Returns
    -------
    np.ndarray, shape (n_values,)
        float64 values in input order.

    Raises
    ------
    ValueError
        If ``values`` is not one-dimensional, holds a value that is not an
        integer or float (booleans and complex included), or holds a NaN or
        infinite value after conversion to float64.
    """
    array = _array(values, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return _finite_array(array, name=name)


def finite_intervals(
    values: ArrayLike, *, name: str, readonly: bool = False
) -> np.ndarray:
    """Return finite float64 ``(n, 2)`` intervals without sorting or merging.

    Empty ``[]`` denotes ``(0, 2)``. Reversed intervals are rejected;
    zero-length intervals are allowed. ``readonly=True`` returns an owned,
    non-writeable copy so the caller's array stays writable.

    Parameters
    ----------
    values : array_like, shape (n_intervals, 2)
        ``[start, stop]`` rows with an integer or float dtype.
    name : str
        Argument name used in the error message.
    readonly : bool, optional
        If True, return a non-writeable copy. Default False.

    Returns
    -------
    np.ndarray, shape (n_intervals, 2)
        float64 intervals in input order.

    Raises
    ------
    ValueError
        If ``values`` does not have shape ``(n, 2)``, holds a value that is
        not an integer or float (booleans and complex included), holds a NaN
        or infinite value, or has a row with ``stop < start``.
    """
    array = _array(values, name=name)
    if array.ndim == 1 and not array.size:
        array = array.reshape(0, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")
    result = _finite_array(array, name=name, readonly=readonly)
    if np.any(result[:, 1] < result[:, 0]):
        raise ValueError(f"{name} stop must be at least its start.")
    return result
