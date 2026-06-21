"""Deterministic, content-addressed identities for v2 selection rows.

Every logical spike-sorting v2 *selection* -- a ``RecordingSelection``,
``ArtifactDetectionSelection``, or ``SortingSelection`` -- must resolve to ONE
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
once duplicated because a ``str`` ``artifact_detection_id`` never compared
equal to the stored ``uuid.UUID``. Every value that has proven dangerous is
normalized to a single representation here so
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

RECORDING_IDENTITY_FIELDS = (
    "nwb_file_name",
    "sort_group_id",
    "interval_list_name",
    "preprocessing_params_name",
    "team_name",
)


def _maybe_uuid(value: str) -> uuid.UUID | None:
    """Return the ``uuid.UUID`` for a UUID-ish string, else ``None``."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None


def _canonical_scalar(value):
    """Normalize one identity value to a single canonical form.

    * ``None`` -> ``None`` (the single "no artifact-detection pass" /
      "absent" form; it JSON-encodes to ``null``, which can never alias a
      UUID string).
    * ``uuid.UUID`` -> canonical lowercase string.
    * a UUID-ish ``str`` -> the same canonical lowercase string, so a
      ``str`` and a ``uuid.UUID`` of the same value share one identity --
      this is the str-vs-UUID ``artifact_detection_id`` bug, fixed at the source.
    * ``bool`` -> kept as ``bool``. ``bool`` is an ``int`` subclass; do
      not collapse ``True``/``False`` into ``1``/``0``.
    * ``int`` and integer-like ids (e.g. a ``numpy`` ``sort_group_id``,
      which implements ``__index__`` but is not an ``int`` subclass) ->
      a plain ``int`` so the numpy and plain forms share one identity.
    * any other ``str`` -> unchanged.

    Raises ``TypeError`` for unsupported value types so a new identity
    field cannot silently serialize to something order- or
    repr-dependent.

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

    Parameters
    ----------
    payload : dict
        The selection's logical-identity fields.

    Returns
    -------
    str
        A byte-stable JSON string with sorted keys and fixed separators.
    """
    normalized = {str(k): _canonical_scalar(v) for k, v in payload.items()}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def deterministic_id(kind: str, payload: dict) -> uuid.UUID:
    """Derive a selection's primary-key UUID from its logical identity.

    ``kind`` ("recording" / "artifact_detection" / "sorting") namespaces
    the three selection tables so identical payloads in different tables
    never alias. Within a table, ``payload`` must carry the FULL logical
    identity -- including an explicit ``source_kind`` for the part-bearing
    tables, whose master row alone does not encode which source produced
    it.

    Parameters
    ----------
    kind : str
        Selection-table namespace: ``"recording"``,
        ``"artifact_detection"``, or ``"sorting"``.
    payload : dict
        The full logical identity of the selection.

    Returns
    -------
    uuid.UUID
        The deterministic ``uuid5`` primary key for the selection.
    """
    return uuid.uuid5(
        V2_SELECTION_NAMESPACE, f"{kind}:{canonical_identity(payload)}"
    )


def recording_identity_payload(key: dict) -> dict:
    """Build a ``RecordingSelection`` logical-identity payload.

    The identity is the full FK set (Raw, SortGroupV2, IntervalList,
    PreprocessingParameters, LabTeam) -- exactly
    :data:`RECORDING_IDENTITY_FIELDS`. The content-addressed
    ``recording_id`` PK is accepted for caller-supplied-id validation but
    does not participate in the payload. Extra fields are rejected rather
    than hashed: passing a joined/fetched dict with non-schema columns must
    not silently produce a different UUID and then fail at ``insert1``.
    Single source of truth shared by ``RecordingSelection.insert_selection``
    and ``preflight_v2_pipeline`` so the two cannot derive different ids for
    the same selection.

    Parameters
    ----------
    key : dict
        The supplied ``RecordingSelection`` fields, including the FK set
        and possibly the ``recording_id`` PK.

    Returns
    -------
    dict
        The canonical FK identity payload, ordered as
        :data:`RECORDING_IDENTITY_FIELDS`.

    Raises
    ------
    ValueError
        If a required identity field is missing, or if ``key`` contains a
        field other than the identity fields and optional ``recording_id``.
    """
    allowed = set(RECORDING_IDENTITY_FIELDS) | {"recording_id"}
    extra = sorted(set(key) - allowed)
    if extra:
        raise ValueError(
            "RecordingSelection.insert_selection received unknown field(s) "
            f"{extra}. Pass only {list(RECORDING_IDENTITY_FIELDS)} and the "
            "optional recording_id; extra joined/fetched columns would change "
            "the deterministic recording_id."
        )
    missing = [field for field in RECORDING_IDENTITY_FIELDS if field not in key]
    if missing:
        raise ValueError(
            "RecordingSelection.insert_selection requires field(s) "
            f"{missing}. Required identity fields are "
            f"{list(RECORDING_IDENTITY_FIELDS)}."
        )
    return {field: key[field] for field in RECORDING_IDENTITY_FIELDS}


def artifact_detection_identity_payload(
    *,
    artifact_detection_params_name,
    recording_id=None,
    shared_artifact_group_name=None,
) -> dict:
    """Build an ``ArtifactDetectionSelection`` logical-identity payload.

    Exactly one of ``recording_id`` (single-recording path) or
    ``shared_artifact_group_name`` (cross-recording path) must be given.
    ``source_kind`` is explicit so a recording source and a shared-group
    source never alias even if their source-identifier strings collide.
    Single source of truth shared by
    ``ArtifactDetectionSelection.insert_selection`` and
    ``preflight_v2_pipeline``.

    Parameters
    ----------
    artifact_detection_params_name : str
        Name of the ``ArtifactDetectionParameters`` row.
    recording_id : optional
        The single-recording source id. Mutually exclusive with
        ``shared_artifact_group_name``. Default ``None``.
    shared_artifact_group_name : optional
        The cross-recording shared-group source name. Mutually exclusive
        with ``recording_id``. Default ``None``.

    Returns
    -------
    dict
        The logical-identity payload, with an explicit ``source_kind``.

    Raises
    ------
    ValueError
        If neither or both of ``recording_id`` and
        ``shared_artifact_group_name`` are given (exactly one source is
        required).
    """
    if (recording_id is None) == (shared_artifact_group_name is None):
        raise ValueError(
            "artifact_detection_identity_payload requires exactly one source: "
            "recording_id xor shared_artifact_group_name."
        )
    if recording_id is not None:
        return {
            "source_kind": "recording",
            "artifact_detection_params_name": artifact_detection_params_name,
            "recording_id": recording_id,
        }
    return {
        "source_kind": "shared_artifact_group",
        "artifact_detection_params_name": artifact_detection_params_name,
        "shared_artifact_group_name": shared_artifact_group_name,
    }


def sorting_identity_payload(
    *,
    recording_id,
    sorter: str,
    sorter_params_name: str,
    artifact_detection_id=None,
) -> dict:
    """Build a ``SortingSelection`` logical-identity payload.

    Identity is the recording source + sorter + the optional artifact
    pass. ``artifact_detection_id`` is normalized to a ``uuid.UUID`` (or
    kept ``None``) so a caller-supplied ``str`` and the stored
    ``uuid.UUID`` share one identity; ``artifact_detection_id=None`` is the
    single "no artifact-detection pass" form and cannot alias any real
    ``artifact_detection_id``. Single source of truth shared by
    ``SortingSelection.insert_selection`` and ``preflight_v2_pipeline``.

    Parameters
    ----------
    recording_id
        The recording source id.
    sorter : str
        Sorter name.
    sorter_params_name : str
        Name of the ``SorterParameters`` row.
    artifact_detection_id : optional
        The optional artifact-detection pass id, normalized to a
        ``uuid.UUID``; ``None`` is the "no artifact-detection pass" form.
        Default ``None``.

    Returns
    -------
    dict
        The logical-identity payload, with ``source_kind`` set to
        ``"recording"``.
    """
    if artifact_detection_id is not None:
        artifact_detection_id = uuid.UUID(str(artifact_detection_id))
    return {
        "source_kind": "recording",
        "recording_id": recording_id,
        "sorter": sorter,
        "sorter_params_name": sorter_params_name,
        "artifact_detection_id": artifact_detection_id,
    }


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

    Parameters
    ----------
    supplied : uuid.UUID, str, or None
        The caller-supplied selection PK. ``None`` is a no-op (the id is
        derived rather than supplied).
    deterministic : uuid.UUID
        The id derived from the selection's logical identity.
    field : str
        Name of the PK field, used in the error message.

    Raises
    ------
    ValueError
        If ``supplied`` is non-``None`` and does not equal
        ``deterministic`` (including when it is not a well-formed UUID).
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
