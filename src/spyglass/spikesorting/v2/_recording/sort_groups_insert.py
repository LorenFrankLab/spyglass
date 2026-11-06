"""Sort-group id allocation and existing-entry checks for ``SortGroupV2``.

DataJoint tables are imported inside the functions.
"""

from __future__ import annotations


def cross_team_downstream(nwb_file_name: str) -> tuple:
    """Per-team downstream rows an overwrite of this session would delete.

    Sort groups in one session can belong to different teams. Visibility
    only; never blocks.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_sel = RecordingSelection & {"nwb_file_name": nwb_file_name}
    summary = []
    for team in sorted(set(rec_sel.fetch("team_name"))):
        team_recs = rec_sel & {"team_name": team}
        sortings = SortingSelection.RecordingSource & team_recs.proj()
        summary.append(
            {
                "team_name": team,
                "recording_selection_rows": len(team_recs),
                "sorting_rows": len(sortings),
                "curation_rows": len(CurationV2 & sortings.proj()),
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

    A rerun is refused unless the caller passes explicit non-overlapping
    ``sort_group_ids`` or ``delete_existing_entries=True, confirm=True``.
    Auto-allocation is allowed only on a session's first call.
    ``table_cls`` is ``SortGroupV2``.
    """
    # Otherwise set() below hides the duplicate until a late DuplicateError.
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

    Starts at 0 on a fresh session. ``table_cls`` is ``SortGroupV2``.
    """
    existing_ids = (table_cls & {"nwb_file_name": nwb_file_name}).fetch(
        "sort_group_id"
    )
    start = int(max(existing_ids)) + 1 if len(existing_ids) else 0
    return list(range(start, start + count))
