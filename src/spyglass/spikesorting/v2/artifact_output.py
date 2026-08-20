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

The single ``dj.Computed`` ``ArtifactDetection`` it supersedes was REMOVED and
split into the two source-specific result tables above; the ``*Output`` suffix
(vs a bare ``*Detection``) matches ``SpikeSortingOutput`` / ``PositionOutput`` /
``LFPOutput``. It stays internal: each result table registers itself here at
materialization (producer-owned) and the sorting stage reads it back;
artifact-removed intervals remain reachable through the per-source result
tables' ``get_artifact_removed_intervals``.
"""

from __future__ import annotations

import datajoint as dj

from spyglass.spikesorting.v2.artifact import (
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
# sibling's table name, so ``_merge_insert``'s ``part_name`` filter stays
# unambiguous. (This merge dispatches inline via the ``&`` checks in
# ``_artifact_source_part_name`` / ``getattr(cls, source)`` below, so it needs
# no module-level source-class map.)


def _artifact_source_part_name(detection_key: dict) -> str:
    """Return the merge part name for a populated artifact-detection key.

    Dispatches on which result table holds the ``artifact_detection_id``.
    Because the id is content-addressed over ``source_kind`` + params +
    source, it lives in exactly one of the two tables.

    Raises
    ------
    KeyError
        If ``detection_key`` is not a populated ``RecordingArtifactDetection``
        or ``SharedGroupArtifactDetection`` (registration into the merge
        follows populate).
    """
    if RecordingArtifactDetection & detection_key:
        return "RecordingSource"
    if SharedGroupArtifactDetection & detection_key:
        return "SharedGroupSource"
    raise KeyError(
        f"{detection_key} is not a populated RecordingArtifactDetection or "
        "SharedGroupArtifactDetection; populate the detection before "
        "registering it into ArtifactDetectionOutput."
    )


@schema
class ArtifactDetectionOutput(_Merge, SpyglassMixin):
    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class RecordingSource(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> RecordingArtifactDetection
        """

    class SharedGroupSource(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> SharedGroupArtifactDetection
        """

    @classmethod
    def insert_detection(
        cls, detection_key: dict, *, skip_duplicates: bool = True
    ) -> None:
        """Idempotently register a populated artifact detection into the merge.

        Parameters
        ----------
        detection_key : dict
            A ``{artifact_detection_id: ...}`` key of a populated
            ``RecordingArtifactDetection`` or ``SharedGroupArtifactDetection``.
        skip_duplicates : bool, optional
            Passed through to ``_merge_insert``; a re-register is a no-op.

        Notes
        -----
        The part name is resolved from which source table holds the id.
        Passing ``part_name`` scopes ``_merge_insert``'s match to the intended
        source, and ``if part & key: continue`` inside ``_merge_insert`` makes
        a re-register a no-op.
        """
        part_name = _artifact_source_part_name(detection_key)
        cls()._merge_insert(
            [detection_key],
            part_name=part_name,
            skip_duplicates=skip_duplicates,
        )

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
    def audit_source_part_integrity(cls) -> list[dict]:
        """Return merge masters whose source-part count is not exactly one.

        A merge master is well-formed iff it has EXACTLY one source part (one
        ``RecordingSource`` xor one ``SharedGroupSource``). The producer-owned
        registration + coordinated deletion keep it that way, but a raw insert
        can create a master with zero parts (orphan), >1 part (ambiguous), or an
        inconsistent ``source`` discriminator. This audit flags all three for a
        maintenance script -- the merge's analogue of
        ``audit_source_part_integrity`` on the XOR selection masters.

        Returns
        -------
        list[dict]
            One entry per offending master: ``merge_id``, the stored ``source``,
            ``source_part_count`` (``0`` = orphan, ``>= 2`` = ambiguous), and
            ``discriminator_ok`` (whether the one part matches ``source``).
            Well-formed masters are omitted.
        """
        flagged: list[dict] = []
        for master in cls.fetch(as_dict=True):
            key = {"merge_id": master["merge_id"]}
            per_part = {
                name: len(getattr(cls, name) & key)
                for name in ("RecordingSource", "SharedGroupSource")
            }
            count = sum(per_part.values())
            discriminator_ok = (
                count == 1 and per_part.get(master["source"], 0) == 1
            )
            if count != 1 or not discriminator_ok:
                flagged.append(
                    {
                        "merge_id": master["merge_id"],
                        "source": master["source"],
                        "source_part_count": count,
                        "discriminator_ok": discriminator_ok,
                    }
                )
        return flagged

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
