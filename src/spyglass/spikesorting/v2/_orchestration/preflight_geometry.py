"""Sort-group contact geometry checks (missing or coincident contacts)."""

from __future__ import annotations

# A preflight message is read in a terminal, so it names enough contacts to
# act on and then says how many it left out. A 128-channel group with no
# geometry would otherwise print 128 triples.
_MAX_REPORTED_CONTACTS = 4


def _truncated(items: list) -> str:
    """Render ``items``, capped, stating how many were omitted."""
    shown = items[:_MAX_REPORTED_CONTACTS]
    omitted = len(items) - len(shown)
    return f"{shown}" + (f" (+{omitted} more)" if omitted else "")


def _missing_coordinate_report(channel_ids, positions) -> str:
    """Name the electrodes with a non-finite coordinate, and which one.

    "electrode 12 is missing rel_z" is actionable; "electrode 12 has no
    position" sends the operator looking for a row that is there. A whole
    unpositioned contact reports all three columns.

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
    import numpy as np

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

    Only called once ``select_distinct_plane`` has returned ``None``, so x-y
    necessarily has at least one duplicate: had it not, ``"xy"`` -- the first
    candidate plane -- would have been chosen.

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
    import numpy as np

    # Same rounding the geometry helpers use to decide "same contact", so the
    # message cannot name a different set than the verdict was based on.
    from spyglass.spikesorting.v2._recording.geometry import _POSITION_DECIMALS

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

    SpikeInterface builds a probe from the contact positions and rejects two
    contacts at the same place, so a sort group whose electrodes share a
    position fails minutes into ``Recording.populate`` (or, on a stored
    artifact, at the analyzer build). This reproduces the *effective* geometry
    the recording stage computes -- ``select_distinct_plane`` over the
    ``Probe.Electrode`` ``rel_x``/``rel_y``/``rel_z``, then the legacy
    ``tetrode_12.5`` repair -- and reports the failure up front instead.

    How much of the geometry is missing is classified by
    :func:`~spyglass.spikesorting.v2._recording.geometry.classify_missing_geometry`
    -- the same predicate ``normalize_channel_locations`` uses at make, so
    this check cannot clear a group that then fails there. NULL / absent
    ``rel_*`` for EVERY coordinate of the whole group is the legacy "geometry
    was never written" case: the raw electrodes table reads back as all-zero,
    which is exactly what the tetrode repair covers, so it is checked as
    all-zero rather than rejected outright. Anything less than that -- one
    NULL ``rel_z`` across the group, or a single unpositioned electrode among
    positioned ones -- is a partially-populated probe and is reported as such,
    naming the electrodes AND the missing columns (``select_distinct_plane``
    would otherwise raise on the NaN rows, and NaNs compare as distinct).

    Checked on the sort group's full electrode membership -- the same list
    ``Recording.make_fetch`` passes to ``maybe_apply_tetrode_geometry``. A
    ``bad_channel_handling='remove'`` run drops members later, which can only
    remove a collision, so this is conservative by at most that case.

    Reads the REGISTERED PROBE's geometry (``Probe.Electrode``, keyed by probe
    type), while the sort itself gets its channel locations from the session's
    own electrodes table. Those agree for any file whose writer filled both
    from one probe definition, but nothing enforces it -- see
    :func:`~spyglass.spikesorting.v2._recording.geometry.fetch_sort_group_contact_positions`.
    A session that disagrees with its probe can therefore pass this check and
    still reach the recording stage's own assertion, which sees what
    SpikeInterface sees.

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
        # ``sort_group_has_electrodes`` owns this failure; reporting it twice
        # would just duplicate the error.
        return None

    registered = fetch_sort_group_contact_positions(nwb_file_name, channel_ids)
    # The same verdict ``normalize_channel_locations`` reaches at make, so a
    # group cannot clear preflight and then fail there.
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
