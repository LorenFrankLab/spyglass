"""Deterministic, content-addressed identities from canonical JSON.

Values are canonicalized so ``uuid.UUID(x)`` and ``str(x)``, or numpy and
plain ints, give the same identity.
"""

from __future__ import annotations

import hashlib
import json
import uuid


def parse_uuid(value) -> uuid.UUID | None:
    """Return the ``uuid.UUID`` for a UUID-ish string, else ``None``."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None


def canonical_scalar(value):
    """Normalize one identity value to a single canonical form.

    - ``None`` -> ``None``.
    - ``uuid.UUID`` or a UUID-ish ``str`` -> canonical lowercase string.
    - ``bool`` -> ``bool`` (not collapsed to ``1``/``0``).
    - ``int`` or integer-like (``__index__``, e.g. numpy) -> plain ``int``.
    - any other ``str`` -> unchanged.

    Raises
    ------
    TypeError
        If ``value`` is not a ``UUID``, ``str``, ``int``, ``bool``,
        ``None``, or an integer-like object implementing ``__index__``.
    """
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        parsed = parse_uuid(value)
        return str(parsed) if parsed is not None else value
    # numpy integers, without importing numpy; float has no __index__.
    if hasattr(value, "__index__"):
        return int(value)
    raise TypeError(
        "identity values must be UUID, str, int, bool, or None; "
        f"got {type(value).__name__!r} ({value!r})"
    )


def canonical_json(payload: dict) -> str:
    """Return a byte-stable JSON string for a logical identity.

    Values are normalized via :func:`canonical_scalar`.

    Parameters
    ----------
    payload : dict
        The logical-identity fields.

    Returns
    -------
    str
        A byte-stable JSON string with sorted keys and fixed separators.
    """
    normalized = {str(k): canonical_scalar(v) for k, v in payload.items()}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def sha256_json(
    payload, *, separators=(",", ":"), allow_nan: bool = True
) -> str:
    """Return the sha256 hex digest of ``payload``'s sorted-key JSON encoding.

    Stored digests depend on the exact bytes, so a caller must keep its
    ``separators`` and ``allow_nan`` while its digests are stored.

    Parameters
    ----------
    payload
        A JSON-serializable value.
    separators : tuple of str, optional
        ``json.dumps`` separators. Default compact ``(",", ":")``;
        ``(", ", ": ")`` reproduces ``json.dumps``'s default spacing.
    allow_nan : bool, optional
        If False, a NaN or infinite float raises ``ValueError`` instead of
        encoding as ``NaN`` / ``Infinity``. Default True.

    Returns
    -------
    str
        The 64-character lowercase sha256 hex digest.

    Raises
    ------
    TypeError
        If ``payload`` holds a value ``json.dumps`` cannot encode.
    ValueError
        If ``allow_nan`` is False and ``payload`` holds a NaN or infinity.
    """
    encoded = json.dumps(
        payload, sort_keys=True, separators=separators, allow_nan=allow_nan
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def deterministic_id(
    namespace: uuid.UUID, kind: str, payload: dict
) -> uuid.UUID:
    """Derive a primary-key UUID from a logical identity.

    Parameters
    ----------
    namespace : uuid.UUID
        The ``uuid5`` namespace of the pipeline that owns the ids.
    kind : str
        Separates tables whose payloads could coincide.
    payload : dict
        The full logical identity.

    Returns
    -------
    uuid.UUID
        ``uuid5(namespace, f"{kind}:{canonical_json(payload)}")``.
    """
    return uuid.uuid5(namespace, f"{kind}:{canonical_json(payload)}")
