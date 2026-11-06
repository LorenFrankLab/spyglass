"""Content fingerprints for v2 parameter-Lookup rows.

A fingerprint identifies a parameter row by its content, excluding the row
name, so two names for the same content collide. Selection identity
(:mod:`_core.selection_identity`) hashes the names instead.

Stdlib-only so a spawned worker that re-imports it opens no DB connection.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._core.selection_identity import sha256_json

_SHORT_FINGERPRINT_LENGTH = 12


def _normalize_numbers(value):
    """Collapse int-valued floats to ints so ``9`` and ``9.0`` canonicalize alike.

    ``extra="allow"`` sorter schemas pass user keys through uncoerced. Bools
    and NaN/Inf pass through unchanged.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _normalize_numbers(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_numbers(item) for item in value]
    return value


def parameter_fingerprint(
    table: str,
    *,
    params: dict,
    params_schema_version: int,
    job_kwargs: dict | None = None,
    sorter: str | None = None,
    matcher: str | None = None,
    execution_params: dict | None = None,
    execution_params_schema_version: int | None = None,
) -> str:
    """Return a SHA-256 content fingerprint for a parameter row.

    The row name is excluded.

    Parameters
    ----------
    table : str
        The Lookup table the row belongs to (e.g.
        ``"PreprocessingParameters"``).
    params : dict
        The schema-validated ``params`` blob.
    params_schema_version : int
        The row's ``params_schema_version``.
    job_kwargs : dict or None, optional
        The per-row ``job_kwargs`` blob.
    sorter : str or None, optional
        For ``SorterParameters`` rows, the ``sorter``; ``None`` elsewhere.
    matcher : str or None, optional
        For ``MatcherParameters`` rows, the ``matcher``; ``None`` elsewhere.
    execution_params : dict or None, optional
        For ``SorterParameters`` rows, the validated ``execution_params`` blob,
        so local and containerized rows are distinct. ``None`` omits it.
    execution_params_schema_version : int or None, optional
        Used only when ``execution_params`` is given.

    Returns
    -------
    str
        The 64-character lowercase hex SHA-256 digest.

    Raises
    ------
    ValueError
        If the payload contains NaN or Inf.
    TypeError
        If the payload is not JSON-encodable.
    """
    payload = {
        "table": table,
        "sorter": sorter,
        "params_schema_version": int(params_schema_version),
        "params": params,
        "job_kwargs": job_kwargs,
    }
    # Optional keys are omitted when unset so other tables' fingerprints are
    # stable.
    if matcher is not None:
        payload["matcher"] = matcher
    if execution_params is not None:
        payload["execution_params"] = execution_params
        payload["execution_params_schema_version"] = int(
            execution_params_schema_version
        )
    return sha256_json(_normalize_numbers(payload), allow_nan=False)


def short_fingerprint(fingerprint: str) -> str:
    """Return a display-length prefix of a digest."""
    return fingerprint[:_SHORT_FINGERPRINT_LENGTH]
