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

import hashlib
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


def shared_group_member_set_hash(recording_ids) -> str:
    """Content-address the member set of a ``SharedArtifactGroup``.

    The sha256 hex digest over the SORTED member ``recording_id`` strings. The
    shared-group artifact identity is ``{params, group_name}`` only, but
    ``ArtifactDetection.make`` scans the LIVE ``SharedArtifactGroup.Member`` set,
    so that set could change under a fixed ``artifact_detection_id``. This hash
    is snapshotted onto ``ArtifactDetectionSelection.SharedGroupSource`` at
    selection time; ``make_fetch`` re-derives it from the live members and
    rejects a drift. Membership is a SET (order-independent), so the ids are
    sorted before hashing -- a member added or removed changes the digest, a
    re-query in a different row order does not.

    Parameters
    ----------
    recording_ids : iterable
        The member ``recording_id`` values (``uuid.UUID`` or ``str``); each is
        stringified, so a freshly-passed str and a DB-fetched UUID agree.

    Returns
    -------
    str
        The 64-char sha256 hex digest.
    """
    ordered = sorted(str(recording_id) for recording_id in recording_ids)
    return hashlib.sha256(
        json.dumps(ordered, sort_keys=True).encode("utf-8")
    ).hexdigest()


def recording_input_hash(
    *,
    electrode_ids,
    reference_mode,
    reference_electrode_id,
    interpolated_bad_channel_ids,
) -> str:
    """Content-address a recording's RESOLVED construction inputs.

    The ``recording_id`` identity is the FK set
    (:data:`RECORDING_IDENTITY_FIELDS`), but ``Recording.make_fetch`` builds the
    preprocessed recording from inputs that live OUTSIDE that FK set and can
    change under a fixed ``sort_group_id`` / ``preprocessing_params_name``:

    * the sort group's electrode MEMBERSHIP (``SortGroupV2.SortGroupElectrode``
      is mutable after downstream recordings exist -- the OP-4 hole);
    * the reference (``reference_mode`` / ``reference_electrode_id``);
    * on the ``interpolate`` bad-channel path, the resolved interior bad-channel
      set (the live ``Electrode.bad_channel='True'`` set -- the OP-3 hole).

    This digest folds those resolved inputs into the ``recording_id`` so a
    changed input mints a NEW recording, and it is snapshotted onto
    ``RecordingSelection.recording_input_hash`` so ``make_fetch`` can re-derive
    it and reject a drift (mirrors ``shared_group_member_set_hash`` /
    ``SharedGroupSource.member_set_hash``). Both id sets are SETS: the ids are
    sorted before hashing, so a re-query in a different row order is stable
    while an added/removed channel changes the digest. On the ``remove`` /
    ``none`` bad-channel paths pass an empty ``interpolated_bad_channel_ids`` --
    those paths re-include nothing, so the bad-channel set does not enter the
    recording's content.

    Parameters
    ----------
    electrode_ids : iterable
        The sort group's member electrode ids (``int`` or integer-like).
    reference_mode : str
        The sort group's ``reference_mode``.
    reference_electrode_id : int or None
        The specific-reference electrode id, or ``None`` for non-specific modes.
    interpolated_bad_channel_ids : iterable
        The resolved interior bad-channel ids re-included on the ``interpolate``
        path; empty on ``remove`` / ``none``.

    Returns
    -------
    str
        The 64-char sha256 hex digest of the canonicalized inputs.
    """
    payload = {
        "electrode_ids": sorted(int(e) for e in electrode_ids),
        "reference_mode": str(reference_mode),
        "reference_electrode_id": (
            None
            if reference_electrode_id is None
            else int(reference_electrode_id)
        ),
        "interpolated_bad_channel_ids": sorted(
            int(c) for c in interpolated_bad_channel_ids
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def sorting_identity_payload(
    *,
    sorter: str,
    sorter_params_name: str,
    recording_id=None,
    concat_recording_id=None,
    artifact_detection_id=None,
) -> dict:
    """Build a ``SortingSelection`` logical-identity payload.

    The sort input is exactly one of a single-session ``recording_id``
    (``source_kind="recording"``) or a ``concat_recording_id``
    (``source_kind="concat"``); ``source_kind`` is part of the identity, so a
    ``recording_id`` and a ``concat_recording_id`` that happen to carry the
    same UUID value can never alias to one ``sorting_id``. The recording
    identity is the recording source + sorter + the optional artifact pass;
    the concat identity is the concat source + sorter (concat sorts have no
    artifact-detection pass, so they fold in no ``artifact_detection_id`` and
    no ``recording_id`` -- neither fabricated nor a ``None`` placeholder).

    ``artifact_detection_id`` is normalized to a ``uuid.UUID`` (or kept
    ``None``) so a caller-supplied ``str`` and the stored ``uuid.UUID`` share
    one identity; ``artifact_detection_id=None`` is the single "no
    artifact-detection pass" form and cannot alias any real
    ``artifact_detection_id``. Single source of truth shared by
    ``SortingSelection.insert_selection`` and ``preflight_v2_pipeline``.

    Parameters
    ----------
    sorter : str
        Sorter name.
    sorter_params_name : str
        Name of the ``SorterParameters`` row.
    recording_id : optional
        The single-session recording source id. Mutually exclusive with
        ``concat_recording_id``. Default ``None``.
    concat_recording_id : optional
        The concatenated-recording source id. Mutually exclusive with
        ``recording_id``. Default ``None``.
    artifact_detection_id : optional
        The optional artifact-detection pass id (recording source only),
        normalized to a ``uuid.UUID``; ``None`` is the "no artifact-detection
        pass" form. Default ``None``.

    Returns
    -------
    dict
        The logical-identity payload, with an explicit ``source_kind``.

    Raises
    ------
    ValueError
        If neither or both of ``recording_id`` and ``concat_recording_id``
        are given (exactly one source is required), or if a concat source is
        combined with an ``artifact_detection_id`` (concat sorts have no
        artifact-detection pass).
    """
    if (recording_id is None) == (concat_recording_id is None):
        raise ValueError(
            "sorting_identity_payload requires exactly one source: "
            "recording_id xor concat_recording_id."
        )
    if concat_recording_id is not None:
        if artifact_detection_id is not None:
            raise ValueError(
                "sorting_identity_payload: a concat source cannot carry an "
                "artifact_detection_id; concat sorts have no artifact pass."
            )
        return {
            "source_kind": "concat",
            "concat_recording_id": concat_recording_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
        }
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
