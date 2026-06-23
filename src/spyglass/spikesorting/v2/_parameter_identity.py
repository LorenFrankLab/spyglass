"""Content fingerprints for v2 parameter-Lookup rows.

A *parameter fingerprint* identifies a ``PreprocessingParameters`` /
``ArtifactDetectionParameters`` / ``SorterParameters`` row by its CONTENT --
the table, the sorter context (for ``SorterParameters``), the schema
version, the validated ``params`` blob, and the per-row ``job_kwargs`` --
with the row NAME deliberately excluded. Two rows with identical content
under different names therefore share a fingerprint; that collision is the
signal the duplicate-content guard and ``describe_parameter_rows`` use.

This is parameter *content* identity, distinct from the selection *logical*
identity in :mod:`_selection_identity` (which hashes a selection's logical
inputs -- including parameter-row *names* -- into a primary-key UUID). The
fingerprint hashes the blob the name points at; the selection identity
hashes the name. They are intentionally separate concerns.

DB-FREE BY CONTRACT. Like :mod:`_selection_identity`, this module imports
neither DataJoint nor SpikeInterface and opens no database connection at
import. Keep it to the standard library so a spawned HPC worker that
re-imports it never triggers a connection.
"""

from __future__ import annotations

import hashlib
import json

_SHORT_FINGERPRINT_LENGTH = 12


def _normalize_numbers(value):
    """Collapse int-valued floats to ints so ``9`` and ``9.0`` canonicalize alike.

    Recurses through dicts and lists. The ``extra="allow"`` sorter schemas
    (Kilosort4, SpykingCircus2, Tridesclous2, generic) pass user keys through
    uncoerced, so a blob may carry ``60000`` under one name and ``60000.0``
    under another for identical science. ``json.dumps`` renders those
    differently, which would fork the
    content fingerprint and defeat the duplicate-content guard. ``bool`` is
    left untouched (``True`` and ``1`` are distinct JSON types and distinct
    intent), and non-integer / NaN / Inf floats (``is_integer()`` is False for
    all of them) pass through so ``allow_nan=False`` can still reject NaN/Inf.
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


def _canonical_content(payload: dict) -> str:
    """Return a byte-stable JSON string for a parameter row's content.

    Keys are sorted recursively and separators are fixed, so the output
    does not depend on dict insertion order or JSON spacing defaults. The
    ``params`` blob is assumed already schema-validated (plain JSON-native
    scalars, lists, and nested dicts). Numbers are normalized
    (:func:`_normalize_numbers`) so an int and an int-valued float fingerprint
    identically -- otherwise two semantically-identical ``extra="allow"`` blobs
    (``60000`` vs ``60000.0``) would fork provenance. ``allow_nan=False`` makes
    a NaN/Inf value raise ``ValueError`` at fingerprint time rather than
    emitting the invalid-JSON tokens ``NaN``/``Infinity`` that no strict JSON
    reader accepts; a non-JSON-serializable value still raises ``TypeError``,
    surfacing an unexpected blob shape rather than being silently coerced.
    """
    return json.dumps(
        _normalize_numbers(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def parameter_fingerprint(
    table: str,
    *,
    params: dict,
    params_schema_version: int,
    job_kwargs: dict | None = None,
    sorter: str | None = None,
    execution_params: dict | None = None,
    execution_params_schema_version: int | None = None,
) -> str:
    """Return a SHA-256 content fingerprint for a parameter row.

    The row NAME is excluded, so two rows whose blobs are identical (same
    table, sorter, schema version, ``params``, ``job_kwargs``, and -- for
    ``SorterParameters`` -- ``execution_params``) produce the same fingerprint
    regardless of what they are called.

    Parameters
    ----------
    table : str
        The Lookup table the row belongs to (e.g.
        ``"PreprocessingParameters"``). Part of the identity so a blob that
        is coincidentally identical across two tables does not collide.
    params : dict
        The schema-validated ``params`` blob.
    params_schema_version : int
        The row's ``params_schema_version``. A blob means something
        different under a different schema version, so it is part of the
        identity. Never bumped to force a new fingerprint -- it is an input
        only.
    job_kwargs : dict or None, optional
        The per-row ``job_kwargs`` blob (concurrency settings). Two rows
        with the same ``params`` but different ``job_kwargs`` are not
        duplicates.
    sorter : str or None, optional
        For ``SorterParameters`` rows, the ``sorter`` the params validate
        against, so duplicate detection is scoped per sorter. ``None`` for
        the single-key Lookup tables.
    execution_params : dict or None, optional
        For ``SorterParameters`` rows, the validated ``execution_params`` blob
        (container backend + install provenance). Included in the identity so a
        local and a containerized row with identical scientific ``params`` are
        NOT duplicates and can coexist under different names. ``None`` (the
        default) omits it entirely, keeping the single-key Lookup tables'
        fingerprints unchanged.
    execution_params_schema_version : int or None, optional
        The row's ``execution_params_schema_version``; only meaningful (and only
        folded into the identity) when ``execution_params`` is provided.

    Returns
    -------
    str
        The 64-character lowercase hex SHA-256 digest of the canonical
        content. Use :func:`short_fingerprint` for a display-length prefix.
    """
    payload = {
        "table": table,
        "sorter": sorter,
        "params_schema_version": int(params_schema_version),
        "params": params,
        "job_kwargs": job_kwargs,
    }
    # Fold execution provenance in ONLY for the tables that carry it
    # (SorterParameters). Omitting the key for the single-key Lookups keeps
    # their fingerprints byte-identical to the pre-execution-params behavior.
    if execution_params is not None:
        payload["execution_params"] = execution_params
        payload["execution_params_schema_version"] = int(
            execution_params_schema_version
        )
    return hashlib.sha256(
        _canonical_content(payload).encode("utf-8")
    ).hexdigest()


def short_fingerprint(
    fingerprint: str, length: int = _SHORT_FINGERPRINT_LENGTH
) -> str:
    """Return the leading ``length`` characters of a full fingerprint.

    A display-friendly prefix for catalogs and error messages; the full
    digest stays available for exact comparison.
    """
    return fingerprint[:length]
