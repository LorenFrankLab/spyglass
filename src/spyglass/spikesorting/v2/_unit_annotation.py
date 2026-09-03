"""Dependency-light normalization for typed unit-annotation sets.

The DataJoint tables live in :mod:`unit_annotation`; this module owns the
canonical value/parameter encoding and hashes so content identity can be
tested without a database connection.  Type tags make ``None``, ``NaN``,
empty text, booleans, integers, and floating-point values unambiguous.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

ANNOTATION_VALUE_TYPES = frozenset({"float", "int", "bool", "text"})
_BIGINT_MIN = -(2**63)
_BIGINT_MAX = 2**63 - 1


def _numpy_scalar_to_python(value):
    """Return a Python scalar for NumPy-like scalar values."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def normalize_parameter_value(value):
    """Normalize a producer parameter into a deterministic JSON-like value."""
    value = _numpy_scalar_to_python(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "+inf" if value > 0 else "-inf"}
        return {"__float__": value.hex()}
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            name = str(key)
            if name in normalized:
                raise ValueError(
                    "producer_parameters contains keys that collide after "
                    f"string normalization: {name!r}."
                )
            normalized[name] = normalize_parameter_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [normalize_parameter_value(item) for item in value]
    raise TypeError(
        "producer_parameters must contain only mappings, lists/tuples, "
        "strings, numbers, booleans, None, or NumPy scalar equivalents; "
        f"got {type(value).__name__}."
    )


def normalize_producer_parameters(parameters) -> dict:
    """Return the normalized mapping stored with an annotation set."""
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise TypeError("producer_parameters must be a mapping or None.")
    return normalize_parameter_value(parameters)


def normalize_annotation_value(value_type: str, value):
    """Validate one runtime value against its declared annotation type."""
    if value_type not in ANNOTATION_VALUE_TYPES:
        raise ValueError(
            f"value_type must be one of {sorted(ANNOTATION_VALUE_TYPES)}; "
            f"got {value_type!r}."
        )
    value = _numpy_scalar_to_python(value)
    if value is None:
        return None
    if value_type == "float":
        if isinstance(value, bool) or not isinstance(value, float):
            raise TypeError(
                "float annotations require float values (integers and "
                f"booleans are not coerced); got {type(value).__name__}."
            )
        return value
    if value_type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                "int annotations require integer values (booleans are not "
                f"coerced); got {type(value).__name__}."
            )
        if not _BIGINT_MIN <= value <= _BIGINT_MAX:
            raise ValueError(
                f"integer annotation {value} is outside MySQL BIGINT range."
            )
        return value
    if value_type == "bool":
        if not isinstance(value, bool):
            raise TypeError(
                "bool annotations require boolean values; got "
                f"{type(value).__name__}."
            )
        return value
    if not isinstance(value, str):
        raise TypeError(
            "text annotations require string values; got "
            f"{type(value).__name__}."
        )
    if len(value) > 255:
        raise ValueError("text annotations are limited to 255 characters.")
    return value


def _encoded_annotation_value(value_type: str, value) -> dict:
    """Encode a validated scalar with an explicit logical type tag."""
    value = normalize_annotation_value(value_type, value)
    if value is None:
        return {"kind": "none"}
    if value_type == "float":
        if math.isnan(value):
            return {"kind": "float", "value": "nan"}
        if math.isinf(value):
            return {
                "kind": "float",
                "value": "+inf" if value > 0 else "-inf",
            }
        return {"kind": "float", "value": value.hex()}
    return {"kind": value_type, "value": value}


def canonical_json_bytes(value) -> bytes:
    """Serialize normalized content with stable separators and key order."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def producer_parameters_hash(parameters) -> str:
    """Hash normalized, self-describing producer parameters."""
    return hashlib.sha256(
        canonical_json_bytes(normalize_producer_parameters(parameters))
    ).hexdigest()


def annotation_set_hash(
    *,
    annotation_name: str,
    annotation_version: int,
    value_type: str,
    producer_parameters,
    values: Sequence[tuple[int, object]],
) -> str:
    """Hash one definition, its normalized parameters, and sorted values."""
    normalized_values = []
    seen = set()
    for unit_id, value in sorted(values, key=lambda item: int(item[0])):
        unit_id = int(unit_id)
        if unit_id in seen:
            raise ValueError(
                f"annotation values contain duplicate unit_id {unit_id}."
            )
        seen.add(unit_id)
        normalized_values.append(
            [unit_id, _encoded_annotation_value(value_type, value)]
        )
    payload = {
        "definition": {
            "name": str(annotation_name),
            "version": int(annotation_version),
            "value_type": str(value_type),
        },
        "producer_parameters": normalize_producer_parameters(
            producer_parameters
        ),
        "values": normalized_values,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


__all__ = [
    "ANNOTATION_VALUE_TYPES",
    "annotation_set_hash",
    "canonical_json_bytes",
    "normalize_annotation_value",
    "normalize_producer_parameters",
    "producer_parameters_hash",
]
