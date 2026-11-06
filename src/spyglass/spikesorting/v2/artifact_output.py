"""The union of artifact-detection sources a sort can mask against.

``ArtifactDetectionOutput`` is a Spyglass merge table over the two
artifact-detection result tables a sort may mask against -- a single-session
``RecordingArtifactDetection`` or a cross-recording
``SharedGroupArtifactDetection`` -- so ``SortingSelection.ArtifactDetectionSource``
carries one foreign key to the merge instead of a pair of mutually-exclusive
source part tables. Like ``LFPOutput`` / ``PositionOutput`` it is an
*input-union* merge (a detection is registered here, after it is populated,
before a ``SortingSelection`` can reference it), not a terminal one like
``SpikeSortingOutput``.

Detection results live in the two source-specific result tables above, not
in one ``dj.Computed`` ``ArtifactDetection`` table; the ``*Output`` suffix
(vs a bare ``*Detection``) matches ``SpikeSortingOutput`` / ``PositionOutput`` /
``LFPOutput``. It stays internal: each result table registers itself here at
materialization (producer-owned) and the sorting stage reads it back;
artifact-removed intervals remain reachable through the per-source result
tables' ``get_artifact_removed_intervals``.

:func:`artifact_lifecycle_locks` is the lock a detection delete and every
selection that links a detection share, and
:func:`insert_artifact_bound_selection` is the selection insert that links one.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager

import datajoint as dj

from spyglass.spikesorting.v2.artifact import (  # noqa: F401 (part FKs)
    RecordingArtifactDetection,
    SharedGroupArtifactDetection,
)
from spyglass.utils import SpyglassMixin, _Merge

schema = dj.schema("spikesorting_v2_artifact_output")

# The merge parts are named ``RecordingSource`` / ``SharedGroupSource`` (NOT the
# source tables' full ``*ArtifactDetection`` names) so the derived part-table +
# FK identifier stays under MySQL's 64-char limit for the long
# ``shared_group_artifact_detection`` source; the two tokens
# (``recording_source`` / ``shared_group_source``) are substrings of neither
# sibling's table name, so ``_Merge``'s source-name-to-part matching stays
# unambiguous.


@schema
class ArtifactDetectionOutput(_Merge, SpyglassMixin):
    """Merge table of the populated artifact detections a sort can mask against.

    Each master row is one registered detection: ``merge_id`` is a uuid that
    ``_merge_insert`` hashes from the source key and ``source``, and ``source``
    names the one part table that holds the detection's
    ``artifact_detection_id``:

    - ``RecordingSource`` -> ``RecordingArtifactDetection`` (one recording);
    - ``SharedGroupSource`` -> ``SharedGroupArtifactDetection`` (one detection
      shared across a group of recordings).

    A detection registers itself here, via :meth:`insert_detection`, in the
    same transaction that materializes its result row; the sort and
    motion-estimate selection inserts register a populated but unregistered
    detection the same idempotent way. Downstream selections reference a
    detection by this table's ``merge_id`` projected as
    ``artifact_detection_merge_id``:
    ``SortingSelection.ArtifactDetectionSource`` (the optional artifact mask
    of a sort) and
    ``MotionEstimateSelection.ArtifactDetectionSource`` (the optional artifact
    mask of a single-recording motion estimate). Use :meth:`get_merge_id` to
    go from a detection to its ``merge_id`` and
    :meth:`resolve_artifact_detection_id` to go back.
    """

    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class RecordingSource(SpyglassMixin, dj.Part):  # noqa: F811
        """Merge part for a single-recording ``RecordingArtifactDetection``."""

        definition = """
        -> master
        ---
        -> RecordingArtifactDetection
        """

    class SharedGroupSource(SpyglassMixin, dj.Part):  # noqa: F811
        """Merge part for a cross-recording ``SharedGroupArtifactDetection``."""

        definition = """
        -> master
        ---
        -> SharedGroupArtifactDetection
        """

    @classmethod
    def insert_detection(cls, detection_key: dict) -> None:
        """Idempotently register a populated artifact detection into the merge.

        Parameters
        ----------
        detection_key : dict
            A ``{artifact_detection_id: ...}`` key of a populated
            ``RecordingArtifactDetection`` or ``SharedGroupArtifactDetection``.

        Raises
        ------
        KeyError
            If ``detection_key`` is not a populated
            ``RecordingArtifactDetection`` or ``SharedGroupArtifactDetection``
            (registration into the merge follows populate).

        Notes
        -----
        ``_merge_insert`` registers the key under the source part whose
        parent table holds the id (the id is content-addressed over its
        source, so exactly one does) and skips a part that already holds it,
        so a re-register is a no-op.
        """
        try:
            cls()._merge_insert([detection_key], skip_duplicates=True)
        except ValueError as exc:
            # Only _merge_insert's "no source table holds the id" error means
            # not populated; any other ValueError (e.g. an ambiguous entry)
            # propagates unchanged.
            if not str(exc).startswith(
                "Non-existing entry in any of the parent tables"
            ):
                raise
            raise KeyError(
                f"{detection_key} is not a populated RecordingArtifactDetection "
                "or SharedGroupArtifactDetection; populate the detection "
                "before registering it into ArtifactDetectionOutput."
            ) from exc

    @classmethod
    def get_merge_id(cls, detection_key: dict):
        """Return the ``merge_id`` of a registered artifact detection.

        Parameters
        ----------
        detection_key : dict
            A ``{artifact_detection_id: ...}`` key.

        Returns
        -------
        uuid.UUID
            The merge id of the (already-registered) detection.

        Raises
        ------
        KeyError
            If the detection is not registered in the merge (zero rows) -- the
            "not registered" signal a caller may catch to mean "no existing
            selection".
        SchemaBypassError
            If more than one row exists for the id in a part -- a corrupt
            duplicate registration (``merge_id`` is a deterministic hash of the
            source key, so >1 is only reachable via a raw insert bypassing
            ``insert_detection``). Raised DISTINCTLY from the not-registered
            ``KeyError`` so a caller's ``except KeyError`` cannot silently
            swallow the corruption.
        """
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        # Fail closed: scan BOTH source parts, not just the first. A well-formed
        # id lives in exactly one; a raw double-insert that registers the same
        # content-addressed id in both parts is corruption that must be surfaced,
        # not silently resolved to whichever part is scanned first.
        matches: list = []
        for part_name in ("RecordingSource", "SharedGroupSource"):
            part = getattr(cls, part_name)
            rows = (part & detection_key).fetch("merge_id")
            if len(rows) > 1:
                raise SchemaBypassError(
                    f"ArtifactDetectionOutput.{part_name} has {len(rows)} rows "
                    f"for {detection_key}; expected at most one -- a corrupt "
                    "duplicate registration (a raw insert bypassing "
                    "insert_detection)."
                )
            if len(rows) == 1:
                matches.append((part_name, rows[0]))
        if len(matches) > 1:
            raise SchemaBypassError(
                f"{detection_key} is registered in >1 source part "
                f"({[m[0] for m in matches]}) of ArtifactDetectionOutput -- a "
                "corrupt cross-part duplicate (a raw insert bypassing "
                "insert_detection). Repair before resolving it."
            )
        if matches:
            return matches[0][1]
        raise KeyError(
            f"{detection_key} is not registered in ArtifactDetectionOutput; "
            "register it via insert_detection first."
        )

    @classmethod
    def resolve_artifact_detection_id(cls, merge_id):
        """Return the per-source ``artifact_detection_id`` behind a merge id.

        The merge_id is exposed only at the ``SortingSelection`` boundary; the
        artifact-id-of-record (folded into ``sorting_id`` and naming the
        ``artifact_detection_{id}`` IntervalList) stays the per-source
        ``artifact_detection_id``. This accessor reads it back through the
        stored ``source`` part.

        Parameters
        ----------
        merge_id : uuid.UUID or str or dict
            A merge id, or a ``{"merge_id": ...}`` restriction.

        Returns
        -------
        uuid.UUID
            The ``artifact_detection_id`` of the registered detection.
        """
        key = merge_id if isinstance(merge_id, dict) else {"merge_id": merge_id}
        source = (cls & key).fetch1("source")
        part = getattr(cls, source)
        return (part & key).fetch1("artifact_detection_id")


@contextmanager
def artifact_lifecycle_locks(detection_ids: Iterable) -> Iterator[None]:
    """Hold the lifecycle lock of every artifact detection in ``detection_ids``.

    The detection ``delete`` and every selection insert that links a detection
    take the same fail-closed advisory lock, keyed on
    ``ArtifactDetectionOutput`` + ``artifact_detection_id``, so a delete
    cannot cascade through a link that is being committed. The locks are
    taken in ascending id order, so two callers locking overlapping sets
    cannot deadlock.

    Parameters
    ----------
    detection_ids : iterable of uuid.UUID or str
        The ``artifact_detection_id`` values to lock; repeats are locked once.
        Empty takes no lock.

    Raises
    ------
    AdvisoryLockError
        If a lock cannot be taken within the lifecycle timeout; locks already
        taken are released.
    """
    from spyglass.spikesorting.v2._core.db_locking import required_advisory_lock

    with ExitStack() as locks:
        for detection_id in sorted({uuid.UUID(str(i)) for i in detection_ids}):
            locks.enter_context(
                required_advisory_lock(
                    ArtifactDetectionOutput,
                    {"artifact_detection_id": detection_id},
                )
            )
        yield


def insert_artifact_bound_selection(
    table_cls,
    pk: dict,
    artifact_detection_id: uuid.UUID | None,
    *,
    insert_rows: Callable[[uuid.UUID | None], None],
    refetch: Callable[[], dict | None],
    consumer: str,
    inputs: str,
    input_tables: str,
) -> dict:
    """Insert a selection that may link an artifact detection, atomically.

    The shared insert of ``SortingSelection`` and ``MotionEstimateSelection``.
    An artifact-bound selection must own its transaction: the lock that
    serializes it against the detection's deletion is released when this
    returns, before a caller's transaction would commit. The detection's
    merge id is resolved (registering a populated but unregistered
    detection) before the transaction, so the transaction holds no merge
    insert and a deterministic-id duplicate collides on the master insert,
    its first statement, with nothing to roll back. A duplicate refetches
    the winner.

    Parameters
    ----------
    table_cls : type
        The selection master class.
    pk : dict
        The master's primary key, returned after a successful insert.
    artifact_detection_id : uuid.UUID or None
        The detection the selection links, or ``None`` for none.
    insert_rows : callable
        Inserts the master first, then its parts, given the detection's
        merge id (``None`` without a detection). Runs inside the
        transaction.
    refetch : callable
        Returns the existing selection's primary key, or ``None``.
    consumer : str
        What the selection is, for messages (e.g. ``"a sort"``).
    inputs : str
        The request's inputs, for the unsatisfied-foreign-key message.
    input_tables : str
        The tables those inputs reference, for the same message.

    Returns
    -------
    dict
        ``pk``, or the existing selection's key after a duplicate.

    Raises
    ------
    ValueError
        If ``artifact_detection_id`` is set inside a caller-owned
        transaction.
    SchemaBypassError
        If the detection is not populated, a duplicate master's parts do not
        match this selection, or an input foreign key is unsatisfied.
    AdvisoryLockError
        If the detection's lifecycle lock cannot be taken.
    """
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    table = table_cls.__name__
    art_merge_id = None
    if artifact_detection_id is not None:
        if table_cls.connection.in_transaction:
            raise ValueError(
                f"{table}.insert_selection: refusing to link an artifact "
                "detection while a caller-owned transaction is open. The "
                "delete-vs-select advisory lock is released when this call "
                "returns -- before your transaction commits -- so it cannot "
                "serialize a concurrent artifact deletion against the "
                "uncommitted selection. Call insert_selection OUTSIDE the "
                "transaction (it manages its own), or link the artifact in a "
                "separate committed step."
            )
        art_key = {"artifact_detection_id": artifact_detection_id}
        try:
            art_merge_id = ArtifactDetectionOutput.get_merge_id(art_key)
        except KeyError:
            try:
                ArtifactDetectionOutput.insert_detection(art_key)
            except KeyError as key_exc:
                raise SchemaBypassError(
                    f"{table}: artifact_detection_id {artifact_detection_id} "
                    "is not materialized in the artifact detection tables (or "
                    "was concurrently deleted). Populate the artifact "
                    f"detection before linking it to {consumer}."
                ) from key_exc
            art_merge_id = ArtifactDetectionOutput.get_merge_id(art_key)

    with artifact_lifecycle_locks(
        [] if artifact_detection_id is None else [artifact_detection_id]
    ):
        try:
            with table_cls._safe_context():
                insert_rows(art_merge_id)
            return pk
        except dj.errors.DuplicateError as exc:
            # A concurrent caller may have inserted the same selection.
            existing = refetch()
            if existing is not None:
                return existing
            master_id = ", ".join(str(value) for value in pk.values())
            raise SchemaBypassError(
                f"{table} master {master_id} exists but its source/artifact "
                "parts do not match this selection: the master was inserted "
                "without insert_selection (a raw-insert orphan). Use "
                "insert_selection(), or drop the orphan master."
            ) from exc
        except dj.errors.IntegrityError as exc:
            raise SchemaBypassError(
                f"{table}: an input foreign key is unsatisfied for {inputs} "
                f"-- the referenced {input_tables} row is missing (a raw "
                "insert bypassing insert_selection, or a concurrent delete "
                "mid-insert). Populate the input, or retry if a transient "
                "race."
            ) from exc
