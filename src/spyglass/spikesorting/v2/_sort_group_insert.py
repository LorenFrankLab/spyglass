"""Sort-group id allocation and existing-entry checks for ``SortGroupV2``.

Both ``SortGroupV2.set_group_by_*`` constructors pick their sort-group ids and
check them against the session's existing rows before inserting:
:func:`next_sort_group_ids` allocates ids after the session's largest, and
:func:`handle_existing` refuses a rerun that would silently extend or
overwrite the session's groups unless the caller confirms a cautious delete.
:func:`cross_team_downstream` counts, per team, the downstream rows such an
overwrite would delete, for ``SortGroupV2.preview_existing_entries``.

Imports without the DB layer: the DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations


def cross_team_downstream(nwb_file_name: str) -> tuple:
    """Per-team downstream rows an overwrite of this session would delete.

    Deleting a session's ``SortGroupV2`` cascades through
    ``RecordingSelection`` (which carries the owning ``team_name``) to
    ``Sorting`` and ``CurationV2``. Because sort groups in one session can
    belong to different teams, this enumerates the cross-team blast radius
    so the operator sees whose downstream rows would vanish before
    confirming. Visibility only -- it never blocks.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_sel = RecordingSelection & {"nwb_file_name": nwb_file_name}
    summary = []
    for team in sorted(set(rec_sel.fetch("team_name"))):
        team_recs = rec_sel & {"team_name": team}
        rec_ids = [str(r) for r in team_recs.fetch("recording_id")]
        sorting_rows = curation_rows = 0
        if rec_ids:
            sortings = SortingSelection.RecordingSource & [
                {"recording_id": r} for r in rec_ids
            ]
            sort_ids = [str(s) for s in sortings.fetch("sorting_id")]
            sorting_rows = len(sortings)
            if sort_ids:
                curation_rows = len(
                    CurationV2 & [{"sorting_id": s} for s in sort_ids]
                )
        summary.append(
            {
                "team_name": team,
                "recording_selection_rows": len(team_recs),
                "sorting_rows": sorting_rows,
                "curation_rows": curation_rows,
            }
        )
    return tuple(summary)


def handle_existing(
    table_cls,
    nwb_file_name: str,
    new_sort_group_ids: list[int],
    explicit_sort_group_ids: bool,
    delete_existing_entries: bool,
    confirm: bool,
) -> None:
    """Enforce the inspect-before-destroy contract.

    Default rerun behavior is to REFUSE: callers must either pass
    explicit non-overlapping ``sort_group_ids`` (an opt-in additive
    insert) or set ``delete_existing_entries=True, confirm=True``
    after reviewing the deletion preview. Auto-allocation is allowed
    ONLY on the first call for a session; on rerun it would silently
    pad ``sort_group_id`` values, which this guard refuses.

    ``table_cls`` is the ``SortGroupV2`` class.
    """
    # Intra-list duplicate check: ``set(new_sort_group_ids)`` below
    # loses the duplicate, and the ``zip`` in the caller would
    # happily build two rows with the same sort_group_id, failing
    # late on a DataJoint duplicate-key error. Catch the typo here
    # for BOTH fresh and existing sessions (auto-allocated ranges
    # are guaranteed unique, so the check only matters when the
    # caller passed an explicit list).
    if explicit_sort_group_ids and len(set(new_sort_group_ids)) != len(
        new_sort_group_ids
    ):
        duplicates = sorted(
            {
                int(s)
                for s in new_sort_group_ids
                if new_sort_group_ids.count(s) > 1
            }
        )
        raise ValueError(
            f"SortGroupV2: sort_group_ids contains duplicate id(s) "
            f"{duplicates}; each sort_group_id can appear at most "
            "once."
        )

    existing = table_cls & {"nwb_file_name": nwb_file_name}
    if len(existing) == 0:
        return

    if not delete_existing_entries:
        if not explicit_sort_group_ids:
            raise ValueError(
                f"SortGroupV2 already has rows for {nwb_file_name!r}; "
                "rerunning without an override would silently extend "
                "the sort-group set. Either pass explicit non-"
                "overlapping sort_group_ids to opt into an additive "
                "insert, or set delete_existing_entries=True, "
                "confirm=True after reviewing "
                f"SortGroupV2.preview_existing_entries({nwb_file_name!r})."
            )
        existing_ids = set(existing.fetch("sort_group_id"))
        overlap = existing_ids & set(new_sort_group_ids)
        if overlap:
            raise ValueError(
                f"SortGroupV2 already has rows for {nwb_file_name!r} "
                f"with overlapping sort_group_ids {sorted(overlap)}. "
                "Pick non-overlapping ids or set "
                "delete_existing_entries=True, confirm=True after "
                f"reviewing SortGroupV2.preview_existing_entries"
                f"({nwb_file_name!r})."
            )
        return

    if not confirm:
        preview = table_cls.preview_existing_entries(nwb_file_name)
        raise ValueError(
            f"delete_existing_entries=True requires confirm=True after "
            f"reviewing the deletion preview. Preview: {preview}. "
            "Re-run with confirm=True to cautiously delete and reinsert."
        )

    existing.cautious_delete()


def next_sort_group_ids(table_cls, nwb_file_name: str, count: int) -> list[int]:
    """Auto-allocate ``count`` sort_group_ids after the session's max.

    Returns the next ``count`` integers starting from
    ``max(existing) + 1`` (or 0 on a fresh session) so additive inserts
    never collide with prior rows on rerun. Shared by both
    ``set_group_by_*`` constructors; ``table_cls`` is the ``SortGroupV2``
    class.
    """
    existing_ids = (table_cls & {"nwb_file_name": nwb_file_name}).fetch(
        "sort_group_id"
    )
    start = int(max(existing_ids)) + 1 if len(existing_ids) else 0
    return list(range(start, start + count))
