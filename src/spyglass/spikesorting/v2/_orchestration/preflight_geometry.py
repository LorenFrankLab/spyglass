"""Sort-group contact geometry checks (missing or coincident contacts)."""

from __future__ import annotations

import numpy as np

from spyglass.spikesorting.v2._recording.geometry import _POSITION_DECIMALS

# Cap terminal output; a 128-channel group would otherwise print 128 triples.
_MAX_REPORTED_CONTACTS = 4


def _truncated(items: list) -> str:
    """Render ``items``, capped, stating how many were omitted."""
    shown = items[:_MAX_REPORTED_CONTACTS]
    omitted = len(items) - len(shown)
    return f"{shown}" + (f" (+{omitted} more)" if omitted else "")


def _missing_coordinate_report(channel_ids, positions) -> str:
    """Name the electrodes with a non-finite coordinate, and which one.

    A whole unpositioned contact reports all three columns.

    Parameters
    ----------
    channel_ids : sequence of int
        Sort-group electrode ids, row-aligned to ``positions``.
    positions : numpy.ndarray
        ``(n, 3)`` contact positions in ``(rel_x, rel_y, rel_z)`` order, at
        least one coordinate of which is non-finite.

    Returns
    -------
    str
        ``"electrode 12 (rel_z); electrode 15 (rel_y, rel_z)"``, capped by
        ``_MAX_REPORTED_CONTACTS``.
    """
    columns = ("rel_x", "rel_y", "rel_z")
    finite = np.isfinite(np.asarray(positions, dtype=float))
    described = []
    for row in np.flatnonzero(~finite.all(axis=1)):
        absent = ", ".join(
            column
            for column, is_finite in zip(columns, finite[row])
            if not is_finite
        )
        described.append(f"electrode {int(channel_ids[row])} ({absent})")
    shown = described[:_MAX_REPORTED_CONTACTS]
    omitted = len(described) - len(shown)
    return "; ".join(shown) + (f" (+{omitted} more)" if omitted else "")


def _coincident_contact_report(channel_ids, positions) -> str:
    """Describe the electrodes that share an x-y position.

    Called only after ``select_distinct_plane`` found no plane, so x-y has a
    duplicate.

    Parameters
    ----------
    channel_ids : sequence of int
        Sort-group electrode ids, row-aligned to ``positions``.
    positions : numpy.ndarray
        ``(n, 3)`` finite contact positions.

    Returns
    -------
    str
        ``"electrodes [1, 2] at (0.0, 0.0); ..."``, capped by
        ``_MAX_REPORTED_CONTACTS`` groups.
    """
    # Same rounding as the verdict, so the message names the same set.
    xy = np.round(np.asarray(positions, dtype=float)[:, :2], _POSITION_DECIMALS)
    groups: dict = {}
    for electrode_id, position in zip(channel_ids, xy):
        groups.setdefault(tuple(position.tolist()), []).append(
            int(electrode_id)
        )
    colliding = [
        (position, ids) for position, ids in groups.items() if len(ids) > 1
    ]
    shown = colliding[:_MAX_REPORTED_CONTACTS]
    report = "; ".join(
        f"electrodes {_truncated(ids)} at {position}" for position, ids in shown
    )
    omitted = len(colliding) - len(shown)
    return report + (f" (+{omitted} more position(s))" if omitted else "")


def sort_group_geometry_problem(
    nwb_file_name: str, sort_group_id: int
) -> "str | None":
    """Report a sort group whose effective 2D contact geometry collapses.

    SpikeInterface rejects coincident contacts. Reproduces the recording
    stage's effective geometry (distinct plane, then the ``tetrode_12.5``
    repair) on the registered ``Probe.Electrode`` positions of every member.
    A group with no positions at all takes the tetrode repair; a partially
    positioned group is reported with its missing columns.

    Parameters
    ----------
    nwb_file_name : str
        The raw NWB file.
    sort_group_id : int
        The sort group to check.

    Returns
    -------
    str or None
        ``None`` when the group's contacts are distinct (or will be made so
        by the tetrode repair); otherwise the operator-facing fix text.
    """
    from spyglass.spikesorting.v2._recording.geometry import (
        effective_contact_plane,
        fetch_sort_group_contact_positions,
        fetch_sort_group_probe_info,
        tetrode_repair_applies,
    )
    from spyglass.spikesorting.v2.recording import SortGroupV2

    channel_ids = sorted(
        (
            SortGroupV2.SortGroupElectrode
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": int(sort_group_id),
            }
        ).fetch("electrode_id"),
        key=int,
    )
    if not channel_ids:
        # ``sort_group_has_electrodes`` reports this.
        return None

    registered = fetch_sort_group_contact_positions(nwb_file_name, channel_ids)
    # Same verdict as ``normalize_channel_locations`` at make.
    verdict, positions, _ = effective_contact_plane(registered)
    if verdict == "partial":
        report = _missing_coordinate_report(channel_ids, positions)
        return (
            f"sort_group_id={int(sort_group_id)} of {nwb_file_name!r} has "
            "electrode(s) with an incomplete Probe.Electrode position "
            f"(missing row or NULL rel_*): {report}. SpikeInterface cannot "
            "place those contacts, and the tetrode_12.5 repair applies only "
            "to a group with NO positions at all. Populate Probe.Electrode "
            "rel_x/rel_y/rel_z for every electrode in the sort group."
        )

    if verdict == "plane":
        return None

    probe_types, electrode_group_names = fetch_sort_group_probe_info(
        nwb_file_name, channel_ids
    )
    if tetrode_repair_applies(
        probe_types, electrode_group_names, len(channel_ids)
    ):
        # The recording stage spreads these onto the 12.5 um square.
        return None
    return (
        f"sort_group_id={int(sort_group_id)} of {nwb_file_name!r} has "
        "contacts that share a position: no coordinate plane separates them. "
        f"Coincident in x-y (the projection SpikeInterface builds its probe "
        f"from): {_coincident_contact_report(channel_ids, positions)}. Fix "
        "Probe.Electrode rel_x/rel_y/rel_z for this sort group's electrodes; "
        "the tetrode_12.5 repair covers only 4-channel single-group tetrodes."
    )
