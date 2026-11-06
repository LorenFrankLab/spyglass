"""Deterministic, content-addressed identities for v2 selection rows.

Each selection's primary key is a ``uuid5`` of its canonical logical identity,
so the primary-key constraint (not a check-then-insert race) deduplicates
concurrent inserts.

Stdlib-only so a spawned worker that re-imports it opens no DB connection.
Values are canonicalized so ``uuid.UUID(x)`` and ``str(x)``, or numpy and
plain ints, give the same identity.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Collection

# uuid5(NAMESPACE_DNS, "spyglass.spikesorting.v2.selection"), frozen as a
# literal.
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
        parsed = _maybe_uuid(value)
        return str(parsed) if parsed is not None else value
    # numpy integers, without importing numpy; float has no __index__.
    if hasattr(value, "__index__"):
        return int(value)
    raise TypeError(
        "selection identity values must be UUID, str, int, bool, or None; "
        f"got {type(value).__name__!r} ({value!r})"
    )


def canonical_identity(payload: dict) -> str:
    """Return a byte-stable JSON string for a selection's logical identity.

    Values are normalized via :func:`_canonical_scalar`.

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


def deterministic_id(kind: str, payload: dict) -> uuid.UUID:
    """Derive a selection's primary-key UUID from its logical identity.

    Payloads for part-bearing tables must include ``source_kind``.

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

    The identity is :data:`RECORDING_IDENTITY_FIELDS`. ``recording_id`` is
    accepted but not hashed. Extra fields are rejected so a joined dict cannot
    silently change the id.

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
    manual_excluded_times=None,
) -> dict:
    """Build an artifact-detection selection logical-identity payload.

    Exactly one of ``recording_id`` or ``shared_artifact_group_name`` must be
    given. ``source_kind`` keeps the two source kinds from aliasing.

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
    manual_excluded_times : optional
        Manual exclusion intervals; hashed into the identity when non-empty.

    Returns
    -------
    dict
        The logical-identity payload, with an explicit ``source_kind``.

    Raises
    ------
    ValueError
        If neither or both of ``recording_id`` and
        ``shared_artifact_group_name`` are given.
    """
    if (recording_id is None) == (shared_artifact_group_name is None):
        raise ValueError(
            "artifact_detection_identity_payload requires exactly one source: "
            "recording_id xor shared_artifact_group_name."
        )
    from spyglass.spikesorting.v2._artifacts.manual import (
        normalize_manual_exclusions,
    )

    exclusions = normalize_manual_exclusions(manual_excluded_times)
    manual = (
        {
            "manual_exclusions_hash": sha256_json(
                exclusions, separators=(", ", ": ")
            )
        }
        if exclusions
        else {}
    )
    if recording_id is not None:
        return {
            **manual,
            "source_kind": "recording",
            "artifact_detection_params_name": artifact_detection_params_name,
            "recording_id": recording_id,
        }
    return {
        **manual,
        "source_kind": "shared_artifact_group",
        "artifact_detection_params_name": artifact_detection_params_name,
        "shared_artifact_group_name": shared_artifact_group_name,
    }


def shared_group_member_set_hash(recording_ids) -> str:
    """Content-address the member set of a ``SharedArtifactGroup``.

    The membership can change under a fixed ``artifact_detection_id``, so the
    hash is stored at selection time and ``make_fetch`` rejects drift.
    Order-independent.

    Parameters
    ----------
    recording_ids : iterable
        The member ``recording_id`` values (``uuid.UUID`` or ``str``).

    Returns
    -------
    str
        The 64-char sha256 hex digest.
    """
    ordered = sorted(str(recording_id) for recording_id in recording_ids)
    return sha256_json(ordered, separators=(", ", ": "))


def recording_input_hash(
    *,
    electrode_ids,
    reference_mode,
    reference_electrode_id,
    interpolated_bad_channel_ids,
) -> str:
    """Content-address a recording's RESOLVED construction inputs.

    Electrode membership, reference, and interpolated bad channels can change
    outside the FK identity. The hash is stored on ``RecordingSelection`` so
    ``make_fetch`` rejects drift. Id sets are order-independent.

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
    return sha256_json(payload)


def sorting_identity_payload(
    *,
    sorter: str,
    sorter_params_name: str,
    recording_id=None,
    concat_recording_id=None,
    artifact_detection_id=None,
    motion_corrected_recording_id=None,
) -> dict:
    """Build a ``SortingSelection`` logical-identity payload.

    ``source_kind`` keeps recording and concat sources from aliasing. Concat
    sources carry no ``artifact_detection_id``. The motion term is omitted, not
    ``None``, for an uncorrected sort.

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
    motion_corrected_recording_id : optional
        The ``MotionCorrectedRecording`` the sort reads instead of its
        source's own traces, for either source kind; ``None`` (the default)
        means the sort reads the source's traces.

    Returns
    -------
    dict
        The logical-identity payload, with an explicit ``source_kind``.

    Raises
    ------
    ValueError
        If neither or both of ``recording_id`` and ``concat_recording_id``
        are given, or if a concat source has an ``artifact_detection_id``.
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
                "artifact_detection_id; configure member masks on the concat selection."
            )
        payload = {
            "source_kind": "concat",
            "concat_recording_id": concat_recording_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
        }
    else:
        if artifact_detection_id is not None:
            artifact_detection_id = uuid.UUID(str(artifact_detection_id))
        payload = {
            "source_kind": "recording",
            "recording_id": recording_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
            "artifact_detection_id": artifact_detection_id,
        }
    if motion_corrected_recording_id is not None:
        payload["motion_corrected_recording_id"] = uuid.UUID(
            str(motion_corrected_recording_id)
        )
    return payload


def reject_unknown_fields(key, allowed, *, caller: str) -> None:
    """Refuse a selection key carrying a field outside ``allowed``.

    Parameters
    ----------
    key : Mapping
        The caller's selection key.
    allowed : set of str
        Every field the selection accepts.
    caller : str
        Names the refusing method in the message.

    Raises
    ------
    ValueError
        If ``key`` has a field not in ``allowed``.
    """
    extra = sorted(set(key) - set(allowed))
    if extra:
        raise ValueError(
            f"{caller} received unknown field(s) {extra}; pass only "
            f"{sorted(allowed)}."
        )


def assert_supplied_id_matches(supplied, deterministic, *, field: str) -> None:
    """Reject a caller-supplied selection PK that is not the deterministic id.

    Parameters
    ----------
    supplied : uuid.UUID, str, or None
        The caller-supplied selection PK. ``None`` is a no-op.
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


def existing_selection_pk(
    master_ids: Collection,
    canonical_id: uuid.UUID,
    *,
    pk_field: str,
    bypass_message: Callable[[list], str],
) -> dict | None:
    """Return the canonical PK among a selection's matching masters, or None.

    Any id other than ``canonical_id`` came from a raw ``insert`` bypass.

    Parameters
    ----------
    master_ids : Collection
        Primary-key values of the matching master rows, shape
        ``(n_matches,)``. Iteration order sets the order of the ids passed to
        ``bypass_message``.
    canonical_id : uuid.UUID
        The deterministic id derived from the selection's logical identity.
    pk_field : str
        Name of the table's primary-key field.
    bypass_message : callable
        Builds the ``DuplicateSelectionError`` message from the list of
        non-deterministic ids.

    Returns
    -------
    dict or None
        ``{pk_field: canonical_id}`` when a master matches, else ``None``.

    Raises
    ------
    DuplicateSelectionError
        If any matching master's id is not ``canonical_id``.
    """
    bypassed = [
        master_id for master_id in master_ids if master_id != canonical_id
    ]
    if bypassed:
        from spyglass.spikesorting.v2.exceptions import DuplicateSelectionError

        raise DuplicateSelectionError(bypass_message(bypassed))
    return {pk_field: canonical_id} if len(master_ids) else None
