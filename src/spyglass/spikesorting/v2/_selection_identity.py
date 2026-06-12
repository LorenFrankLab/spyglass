"""Deterministic, content-addressed identities for v2 selection rows.

Every logical spike-sorting v2 *selection* -- a ``RecordingSelection``,
``ArtifactSelection``, or ``SortingSelection`` -- must resolve to ONE
stable primary-key UUID under serial, repeated, concurrent, and
worker-retry insertion. This module derives that UUID from the
selection's canonical logical identity with :func:`uuid.uuid5`, so the
primary-key uniqueness constraint -- not a check-then-insert dedup race --
becomes the concurrency guard. Two callers that ask for the same logical
selection compute the same id; the database accepts one master row and
rejects the duplicate, and the loser refetches the winner's row.

DB-FREE BY CONTRACT. Like ``_artifact_compute``, this module imports
neither DataJoint nor SpikeInterface and opens no database connection at
import. The selection helpers compute a primary key here BEFORE touching
the DB, and an HPC job array re-importing this module in a spawned worker
(macOS ``spawn`` re-imports the defining module) must never trigger a
connection. Keep it to the standard library.

Canonicalization is the footgun this module exists to kill: a v2 sort was
once duplicated because a ``str`` ``artifact_id`` never compared equal to
the stored ``uuid.UUID`` (see
``test_insert_selection_dedup_accepts_str_artifact_id``). Every value that
has proven dangerous is normalized to a single representation here so
``uuid.UUID(x)`` and ``str(x)`` -- and a ``numpy`` vs a plain
``sort_group_id`` -- produce the SAME identity.
"""

from __future__ import annotations

import json
import uuid

# Fixed namespace for every v2 selection UUIDv5. Derived once as
#   uuid.uuid5(uuid.NAMESPACE_DNS, "spyglass.spikesorting.v2.selection")
# and then frozen as a literal so the value can never drift if the seed
# string is later edited. uuid5 is a pure hash (no randomness), so this is
# stable across processes, machines, and Python versions.
V2_SELECTION_NAMESPACE = uuid.UUID("b44d4765-4714-5c69-96d5-97feb2217e86")


def _maybe_uuid(value: str) -> uuid.UUID | None:
    """Return the ``uuid.UUID`` for a UUID-ish string, else ``None``."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None


def _canonical_scalar(value):
    """Normalize one identity value to a single canonical form.

    * ``None`` -> ``None`` (the single "no artifact" / "absent" form; it
      JSON-encodes to ``null``, which can never alias a UUID string).
    * ``uuid.UUID`` -> canonical lowercase string.
    * a UUID-ish ``str`` -> the same canonical lowercase string, so a
      ``str`` and a ``uuid.UUID`` of the same value share one identity --
      this is the str-vs-UUID ``artifact_id`` bug, fixed at the source.
    * ``bool`` -> kept as ``bool``. ``bool`` is an ``int`` subclass; do
      not collapse ``True``/``False`` into ``1``/``0``.
    * ``int`` and integer-like ids (e.g. a ``numpy`` ``sort_group_id``,
      which implements ``__index__`` but is not an ``int`` subclass) ->
      a plain ``int`` so the numpy and plain forms share one identity.
    * any other ``str`` -> unchanged.

    Raises ``TypeError`` for unsupported value types so a new identity
    field cannot silently serialize to something order- or
    repr-dependent.
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
        parsed = _maybe_uuid(value)
        return str(parsed) if parsed is not None else value
    # numpy integers (np.int64, ...) are integer-like but not ``int``
    # subclasses; they implement ``__index__``. Collapse them to a plain
    # ``int`` -- without importing numpy, keeping this module dependency
    # light -- so a numpy ``sort_group_id`` matches the int the table
    # stores. (``float`` has no ``__index__``, so it still raises below.)
    if hasattr(value, "__index__"):
        return int(value)
    raise TypeError(
        "selection identity values must be UUID, str, int, bool, or None; "
        f"got {type(value).__name__!r} ({value!r})"
    )


def canonical_identity(payload: dict) -> str:
    """Return a byte-stable JSON string for a selection's logical identity.

    Keys are sorted and separators are fixed so the output does not depend
    on dict insertion order or Python's JSON spacing defaults; every value
    is normalized via :func:`_canonical_scalar` so equivalent inputs
    collapse to one identity.
    """
    normalized = {str(k): _canonical_scalar(v) for k, v in payload.items()}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def deterministic_id(kind: str, payload: dict) -> uuid.UUID:
    """Derive a selection's primary-key UUID from its logical identity.

    ``kind`` ("recording" / "artifact" / "sorting") namespaces the three
    selection tables so identical payloads in different tables never
    alias. Within a table, ``payload`` must carry the FULL logical
    identity -- including an explicit ``source_kind`` for the part-bearing
    tables, whose master row alone does not encode which source produced
    it.
    """
    return uuid.uuid5(
        V2_SELECTION_NAMESPACE, f"{kind}:{canonical_identity(payload)}"
    )


def assert_supplied_id_matches(supplied, deterministic, *, field: str) -> None:
    """Reject a caller-supplied selection PK that is not the deterministic id.

    The selection helpers derive each PK from the logical identity, so a
    caller normally omits it (``supplied is None`` -> no-op). If a caller
    DOES pass one, it must equal the deterministic id; a mismatch means a
    hand-rolled random / legacy UUID that would silently fork the
    selection identity, so raise rather than honor it. Accepts ``str`` or
    ``uuid.UUID`` for ``supplied`` (normalized before comparison). A
    ``supplied`` value that is not even a well-formed UUID is, by
    definition, not the deterministic id, so it raises the SAME curated
    message rather than a low-level ``uuid.UUID`` parse error.
    """
    if supplied is None:
        return
    if isinstance(supplied, uuid.UUID):
        normalized = supplied
    elif isinstance(supplied, str):
        normalized = _maybe_uuid(supplied)  # None if not a well-formed UUID
    else:
        normalized = None
    if normalized != deterministic:
        raise ValueError(
            f"insert_selection: supplied {field}={supplied!r} does not match "
            f"the deterministic id {deterministic} derived from the "
            f"selection's logical identity. Omit {field}; the id is "
            "content-addressed from the logical fields."
        )
