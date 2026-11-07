"""Channel-id / probe-geometry services behind ``Recording``.

Only the ``fetch_*`` functions read the database.
"""

from __future__ import annotations


def spikeinterface_channel_ids(nwb_file_abs_path: str, spyglass_ids):
    """Map Spyglass electrode_ids onto SpikeInterface channel ids.

    SpikeInterface 0.104 uses the electrodes table's ``channel_name`` column
    as the channel id when present, else ``electrode_id``. ``channel_name``
    is positional and electrode ids need not equal row positions, so each id
    is resolved to its row first.

    Parameters
    ----------
    nwb_file_abs_path : str
        Absolute path of the parent (raw) NWB whose electrodes table is read.
    spyglass_ids : sequence of int
        Spyglass electrode ids to map onto SI channel ids.

    Returns
    -------
    list
        SpikeInterface channel ids in ``spyglass_ids`` order:
        ``channel_name`` strings when the electrodes table has that column,
        otherwise the integer electrode ids.

    Raises
    ------
    ValueError
        If any requested electrode id is not in the electrodes table, or maps
        to more than one row (ambiguous id).
    """
    from pathlib import Path

    import pynwb

    from spyglass.utils.nwb_helper_fn import (
        get_electrode_indices,
        invalid_electrode_index,
    )

    ids = [int(c) for c in spyglass_ids]
    nwb_file_name = Path(nwb_file_abs_path).name
    with pynwb.NWBHDF5IO(nwb_file_abs_path, mode="r") as io:
        nwbfile = io.read()
        electrodes_table = nwbfile.electrodes
        table_ids = [int(e) for e in electrodes_table.id[:]]
        row_indices = get_electrode_indices(nwbfile, ids)
        missing = [
            eid
            for eid, idx in zip(ids, row_indices)
            if idx == invalid_electrode_index
        ]
        if missing:
            raise ValueError(
                f"spikeinterface_channel_ids: electrode ids {missing} are not "
                f"in the {nwb_file_name!r} electrodes table"
            )
        ambiguous = sorted({eid for eid in ids if table_ids.count(eid) > 1})
        if ambiguous:
            raise ValueError(
                f"spikeinterface_channel_ids: electrode ids {ambiguous} map to "
                f"more than one row in the {nwb_file_name!r} electrodes table"
            )
        if "channel_name" not in electrodes_table.colnames:
            return ids
        channel_names = electrodes_table["channel_name"]
        return [channel_names[idx] for idx in row_indices]


def fetch_sort_group_probe_info(
    nwb_file_name: str, channel_ids
) -> tuple[tuple, tuple]:
    """Fetch per-channel ``probe_type`` + ``electrode_group_name``.

    Ordered by ``electrode_id`` so DataJoint's two ``make_fetch`` calls
    return identical tuples.

    Parameters
    ----------
    nwb_file_name : str
        Parent NWB filename to restrict the ``Electrode * Probe``
        fetch.
    channel_ids : sequence of int
        Spyglass electrode ids to look up, one metadata entry per id.

    Returns
    -------
    probe_types : tuple
        ``probe_type`` per channel id, ordered by ``electrode_id``.
    electrode_group_names : tuple
        ``electrode_group_name`` per channel id, ordered by
        ``electrode_id``.
    """
    from spyglass.common.common_device import Probe as _Probe
    from spyglass.common.common_ephys import Electrode as _Electrode

    probe_rows = (
        _Electrode * _Probe
        & {"nwb_file_name": nwb_file_name}
        & [{"electrode_id": int(c)} for c in channel_ids]
    ).fetch(
        "probe_type",
        "electrode_group_name",
        as_dict=True,
        order_by="electrode_id",
    )
    probe_types = tuple(r["probe_type"] for r in probe_rows)
    electrode_group_names = tuple(r["electrode_group_name"] for r in probe_rows)
    return probe_types, electrode_group_names


def fetch_sort_group_contact_positions(nwb_file_name: str, channel_ids):
    """Fetch the ``Probe.Electrode`` 3D contact position of each channel.

    The probe-relative ``rel_*`` columns, not ``Electrode.x/y/z``. This is
    the registered probe type's geometry, which can differ from the
    session's electrodes table that SpikeInterface reads. Missing positions
    arrive as ``NaN``, not 0.0.

    Parameters
    ----------
    nwb_file_name : str
        Parent NWB filename restricting the fetch.
    channel_ids : sequence of int
        Spyglass electrode ids to look up.

    Returns
    -------
    numpy.ndarray
        ``(len(channel_ids), 3)`` float array of ``(rel_x, rel_y, rel_z)``,
        row-aligned to ``sorted(channel_ids)``, as
        :func:`fetch_sort_group_probe_info` is. ``NaN`` where the electrode
        has no probe link or a NULL coordinate.
    """
    import numpy as np

    from spyglass.common.common_device import Probe as _Probe
    from spyglass.common.common_ephys import Electrode as _Electrode

    wanted = sorted(int(c) for c in channel_ids)
    rows = (
        (_Electrode * _Probe.Electrode)
        & {"nwb_file_name": nwb_file_name}
        & [{"electrode_id": c} for c in wanted]
    ).fetch("electrode_id", "rel_x", "rel_y", "rel_z", as_dict=True)
    by_id = {int(r["electrode_id"]): r for r in rows}
    positions = np.full((len(wanted), 3), np.nan, dtype=float)
    for index, electrode_id in enumerate(wanted):
        row = by_id.get(electrode_id)
        if row is None:
            continue
        for axis, column in enumerate(("rel_x", "rel_y", "rel_z")):
            if row[column] is not None:
                positions[index, axis] = float(row[column])
    return positions


def _tetrode_repair_skip_reason(
    probe_types: tuple, electrode_group_names: tuple, n_channels: int
) -> "str | None":
    """Why the ``tetrode_12.5`` repair does NOT apply, or ``None`` if it does.

    Shared with preflight via :func:`tetrode_repair_applies`.
    """
    unique_probes = set(probe_types)
    unique_groups = set(electrode_group_names)
    # next(iter(...), None) avoids StopIteration on an empty probe set.
    gates = (
        (
            len(unique_probes) != 1,
            "sort group spans multiple probe types "
            "(expected a single tetrode_12.5)",
        ),
        (
            next(iter(unique_probes), None) != "tetrode_12.5",
            "single probe is not tetrode_12.5",
        ),
        (
            int(n_channels) != 4,
            "sort group does not have exactly 4 channels",
        ),
        (
            len(unique_groups) != 1,
            "sort group spans multiple electrode groups",
        ),
    )
    for failed, reason in gates:
        if failed:
            return reason
    return None


def tetrode_repair_applies(
    probe_types: tuple, electrode_group_names: tuple, n_channels: int
) -> bool:
    """True when :func:`maybe_apply_tetrode_geometry` would spread this group.

    Preflight uses this so a legacy all-zero tetrode is not reported broken.

    Parameters
    ----------
    probe_types : tuple
        ``probe_type`` per channel in the sort group.
    electrode_group_names : tuple
        ``electrode_group_name`` per channel in the sort group.
    n_channels : int
        Number of channels in the sort group.

    Returns
    -------
    bool
        True when every gate passes.
    """
    return (
        _tetrode_repair_skip_reason(
            probe_types, electrode_group_names, n_channels
        )
        is None
    )


def maybe_apply_tetrode_geometry(
    recording,
    probe_types: tuple,
    electrode_group_names: tuple,
    sort_group_channel_ids: list,
):
    """Attach the ``tetrode_12.5`` probe geometry when the sort group fits.

    Exactly 4 channels on one ``tetrode_12.5`` probe and one electrode
    group get the 12.5 µm square (6.25 µm radius), replacing any stored
    positions. For legacy Frank-lab NWBs without contact positions. A
    skipped repair logs the failed gate at INFO.

    Parameters
    ----------
    recording : si.BaseRecording
        The sliced recording to attach probe geometry to.
    probe_types : tuple
        ``probe_type`` per channel; the patch requires a single
        ``"tetrode_12.5"`` value across the group.
    electrode_group_names : tuple
        ``electrode_group_name`` per channel; the patch requires a
        single electrode group across the group.
    sort_group_channel_ids : list
        Spyglass electrode ids of the sort group; the patch requires
        exactly four.

    Returns
    -------
    si.BaseRecording
        The recording with tetrode geometry attached when every gate
        passes, otherwise the input recording unchanged.
    """
    from spyglass.utils import logger

    reason = _tetrode_repair_skip_reason(
        probe_types, electrode_group_names, len(sort_group_channel_ids)
    )
    if reason is not None:
        logger.info("maybe_apply_tetrode_geometry skipped: %s", reason)
        return recording

    import numpy as np
    import probeinterface as pi

    tetrode = pi.Probe(ndim=2)
    tetrode.set_contacts(
        [list(p) for p in TETRODE_12_5_POSITIONS],
        shapes="circle",
        shape_params={"radius": 6.25},
    )
    tetrode.set_contact_ids([str(c) for c in sort_group_channel_ids])
    tetrode.set_device_channel_indices(np.arange(4))
    return recording.set_probe(tetrode, in_place=True)


# Contact positions (um) the ``tetrode_12.5`` repair installs, in sort-group
# channel order.
TETRODE_12_5_POSITIONS = ((0.0, 0.0), (0.0, 12.5), (12.5, 0.0), (12.5, 12.5))


def sort_group_planar_positions(positions, probe_types, electrode_group_names):
    """Return the 2D contact positions the recording stage gives a sort group.

    The ``tetrode_12.5`` square when the repair applies, else the plane that
    separates every contact.

    Parameters
    ----------
    positions : array_like
        ``(n_contacts, 3)`` ``(rel_x, rel_y, rel_z)`` positions in sorted
        electrode-id order, NaN where missing.
    probe_types, electrode_group_names : tuple
        Per-channel metadata, as :func:`fetch_sort_group_probe_info` returns.

    Returns
    -------
    source : {"tetrode_12.5", "xy", "xz", "yz"} or None
        The repair, or the chosen axis pair; ``None`` when no plane
        separates every contact or a position is missing (preflight refuses
        such a group).
    planar : numpy.ndarray or None
        ``(n_contacts, 2)`` positions, or ``None`` with ``source``.
    """
    import numpy as np

    if tetrode_repair_applies(
        tuple(probe_types), tuple(electrode_group_names), len(positions)
    ):
        return "tetrode_12.5", np.asarray(TETRODE_12_5_POSITIONS)
    verdict, registered, _ = effective_contact_plane(positions)
    if verdict != "plane":
        return None, None
    return select_distinct_plane(registered)


# x-y first: probeinterface's select_axes default.
_CANDIDATE_PLANES = ("xy", "xz", "yz")

# µm; below any contact pitch, above float32 noise of the rel_* columns.
_POSITION_DECIMALS = 6


def classify_missing_geometry(locations) -> str:
    """Say how much of a contact-position set is missing.

    Preflight and the recording stage must agree on this.

    - ``"complete"``: no coordinate is finite; treated as the legacy
      all-zero geometry the ``tetrode_12.5`` repair covers.
    - ``"partial"``: some coordinates are non-finite; nothing repairs this.
    - ``"none"``: every coordinate is finite, including all-zero.

    An empty set raises: ``"complete"`` would wrongly route it to the
    all-zero geometry.

    Parameters
    ----------
    locations : array_like
        ``(n_contacts, 3)`` contact positions in ``(rel_x, rel_y, rel_z)``
        order, with at least one contact.

    Returns
    -------
    str
        ``"complete"``, ``"partial"`` or ``"none"``.

    Raises
    ------
    ValueError
        If ``locations`` holds no contacts.
    """
    import numpy as np

    positions = np.asarray(locations, dtype=float)
    if positions.size == 0:
        raise ValueError(
            "classify_missing_geometry: no contacts to classify (got shape "
            f"{positions.shape}). A sort group with no electrodes is a "
            "membership problem, not a geometry one."
        )
    finite = np.isfinite(positions)
    if not finite.any():
        return "complete"
    if not finite.all():
        return "partial"
    return "none"


def _all_rows_distinct(positions) -> bool:
    """True when no two rows of ``positions`` coincide (to 6 decimals)."""
    import numpy as np

    rounded = np.round(np.asarray(positions, dtype=float), _POSITION_DECIMALS)
    return len(np.unique(rounded, axis=0)) == len(rounded)


def assert_finite_contact_positions(locations) -> None:
    """Refuse contact positions with a non-finite coordinate.

    NaN rows compare as distinct under ``np.unique``, so they would pass
    :func:`assert_unique_contact_positions`.

    Parameters
    ----------
    locations : array_like
        ``(n_contacts, 3)`` contact positions.

    Raises
    ------
    ValueError
        If any coordinate is non-finite, naming the offending rows.
    """
    import numpy as np

    loc = np.asarray(locations, dtype=float)
    if np.isfinite(loc).all():
        return
    bad = np.flatnonzero(~np.isfinite(loc).all(axis=1))
    raise ValueError(
        "Recording geometry: contact positions must be finite, but "
        f"row(s) {bad.tolist()} are {loc[bad].tolist()}. Populate "
        "Probe.Electrode rel_x/rel_y/rel_z for this sort group."
    )


def select_distinct_plane(locations):
    """Pick the axis pair in which every contact has a distinct position.

    SpikeInterface projects to x-y when building a probe and rejects
    coincident contacts. Frank-lab tetrodes lie in the x-z plane.

    Parameters
    ----------
    locations : array_like
        ``(n_contacts, 3)`` contact positions, in the ``Probe.Electrode``
        ``(rel_x, rel_y, rel_z)`` order.

    Returns
    -------
    tuple of (str, numpy.ndarray) or None
        The chosen axis pair (``"xy"``, ``"xz"`` or ``"yz"``) and the
        corresponding ``(n_contacts, 2)`` positions, preferring ``"xy"``.
        ``None`` when no pair separates every contact.

    Raises
    ------
    ValueError
        If ``locations`` is not ``(n_contacts, 3)``, or any coordinate is
        non-finite.
    """
    import numpy as np
    from probeinterface import select_axes

    loc = np.asarray(locations, dtype=float)
    if loc.ndim != 2 or loc.shape[1] != 3:
        raise ValueError(
            "select_distinct_plane: expected an (n_contacts, 3) array of "
            f"contact positions, got shape {loc.shape}"
        )
    assert_finite_contact_positions(loc)
    for axes in _CANDIDATE_PLANES:
        positions = select_axes(loc, axes)
        if _all_rows_distinct(positions):
            return axes, positions
    return None


def effective_contact_plane(locations) -> tuple:
    """Classify a sort group's registered positions the way preflight does.

    :func:`classify_missing_geometry` then :func:`select_distinct_plane`,
    reading no positions as all-zero. The ``tetrode_12.5`` rescue is left
    to the caller.

    Parameters
    ----------
    locations : array_like
        ``(n_contacts, 3)`` ``(rel_x, rel_y, rel_z)`` positions, NaN where
        missing, with at least one contact.

    Returns
    -------
    verdict : {"partial", "plane", "collapsed"}
        ``"partial"``: some but not all coordinates are missing.
        ``"plane"``: an axis pair separates every contact. ``"collapsed"``:
        none does (including a group with no positions at all).
    positions : numpy.ndarray
        ``(n_contacts, 3)`` positions, all zero when none was registered.
    planar : numpy.ndarray or None
        ``(n_contacts, 2)`` positions in the chosen plane for ``"plane"``.
    """
    import numpy as np

    positions = np.asarray(locations, dtype=float)
    missing = classify_missing_geometry(positions)
    if missing == "partial":
        return "partial", positions, None
    if missing == "complete":
        positions = np.zeros_like(positions)
    chosen = select_distinct_plane(positions)
    if chosen is None:
        return "collapsed", positions, None
    return "plane", positions, chosen[1]


def retained_channel_ids(
    channel_ids,
    *,
    reference_mode: str,
    reference_electrode_id,
    bad_channel_handling: str,
    bad_channel_ids=(),
) -> list[int]:
    """The contacts the preprocessed recording keeps, for choosing its plane.

    Interpolated bad channels stay; a ``specific`` reference is dropped,
    since it may sit on another probe of the same type and duplicate a
    member's position.

    Parameters
    ----------
    channel_ids : sequence of int
        Sorted electrode ids for the sort group.
    reference_mode : str
        Referencing mode (e.g. ``'none'``, ``'specific'``).
    reference_electrode_id : int or None
        Reference electrode id; non-``None`` only for ``'specific'``.
    bad_channel_handling : str
        ``'interpolate'`` or ``'remove'``.
    bad_channel_ids : sequence of int, optional
        Interior bad channels re-included on the ``interpolate`` path.

    Returns
    -------
    list[int]
        Sorted retained electrode ids.
    """
    retained = {int(c) for c in channel_ids}
    if bad_channel_handling == "interpolate":
        retained |= {int(c) for c in bad_channel_ids}
    if reference_mode == "specific":
        retained -= {int(reference_electrode_id)}
    return sorted(retained)


def normalize_channel_locations(recording, *, channel_ids=None):
    """Reduce 3D channel locations to the distinct 2D plane, in place.

    Must run before any probe is attached: SpikeInterface refuses to set
    locations once a ``contact_vector`` exists.

    - Returned untouched when already 2D, without locations, when no plane
      separates the contacts, or when every coordinate is non-finite (the
      case preflight routes to the ``tetrode_12.5`` repair).
    - Partially non-finite positions raise.
    - ``channel_ids`` scopes only the plane choice; the projection is
      applied to every channel. This keeps a ``specific`` reference on a
      same-type probe from blocking the plane.

    Parameters
    ----------
    recording : si.BaseRecording
        The channel-sliced recording, with no probe attached.
    channel_ids : sequence, optional
        Keyword-only. Channel ids of the contacts the plane is chosen from;
        ``None`` (default) uses every channel. Pass every retained contact,
        i.e. all but the ``specific`` reference.

    Returns
    -------
    si.BaseRecording
        The same recording, with 2D channel locations when a plane was chosen.

    Raises
    ------
    ValueError
        If a probe is already attached to ``recording``, if ``channel_ids``
        names a channel the recording does not carry, or if some but not all
        3D contact positions on any sliced channel are non-finite.
    """
    import numpy as np
    from probeinterface import select_axes

    from spyglass.utils import logger

    if recording.get_property("contact_vector") is not None:
        raise ValueError(
            "normalize_channel_locations: a probe is already attached to this "
            "recording, so its channel locations can no longer be rewritten. "
            "Normalize the geometry before any probe is built (before "
            "set_probe / get_probe / create_sorting_analyzer)."
        )
    # get_channel_locations raises without the property, so test it directly.
    locations = recording.get_property("location")
    if locations is None or np.asarray(locations).shape[1] != 3:
        return recording

    positions_3d = np.asarray(
        recording.get_channel_locations(axes="xyz"), dtype=float
    )
    # ids_to_indices raises naming any unknown ids.
    retained = (
        positions_3d
        if channel_ids is None
        else positions_3d[recording.ids_to_indices(list(channel_ids))]
    )
    # Missing geometry is judged on the whole sliced set.
    if classify_missing_geometry(positions_3d) == "complete":
        return recording
    # Interpolation's kriging weights use every channel's location.
    assert_finite_contact_positions(positions_3d)

    chosen = select_distinct_plane(retained)
    if chosen is None:
        return recording
    axes, _ = chosen
    if axes != "xy":
        logger.info(
            "normalize_channel_locations: contacts are not distinct in x-y; "
            "using the %s plane for this sort group",
            axes,
        )
    recording.set_channel_locations(select_axes(positions_3d, axes))
    return recording


def assert_unique_contact_positions(
    recording, *, require_2d: bool = True
) -> None:
    """Require every contact to have a distinct 2D position.

    Checks ``get_channel_locations()``, the geometry
    ``create_sorting_analyzer`` will see, so run it last. With
    ``require_2d``, 3D locations are refused here rather than later in the
    artifact writer.

    Parameters
    ----------
    recording : si.BaseRecording
        The fully prepared recording, after normalization and any probe patch.
    require_2d : bool, default True
        Keyword-only. When False, 3D locations are projected to x-y instead
        of refused; reloaded artifacts always come back 3D.

    Raises
    ------
    ValueError
        If the recording carries no geometry at all, if ``require_2d`` and its
        locations are still 3D, or if two or more contacts share a 2D position.
    """
    import numpy as np

    # get_channel_locations would raise a bare Exception here.
    if (
        recording.get_property("contact_vector") is None
        and recording.get_property("location") is None
    ):
        raise ValueError(
            "Recording.make: this recording carries no contact positions at "
            "all. Populate Probe.Electrode rel_x/rel_y/rel_z for this sort "
            "group's electrodes."
        )
    if (
        require_2d
        and recording.get_property("location") is not None
        and recording.has_3d_locations()
    ):
        raise ValueError(
            "Recording.make: this recording still carries 3D channel "
            "locations; normalize them to a 2D plane before the artifact is "
            "written (normalize_channel_locations). Check Probe.Electrode "
            "rel_x/rel_y/rel_z for this sort group -- a reference channel "
            "that takes part in plane selection and is dropped afterwards can "
            "leave the group unnormalized."
        )
    positions = np.asarray(recording.get_channel_locations(), dtype=float)
    if len(positions) > 1 and not _all_rows_distinct(positions):
        raise ValueError(
            "Recording.make: contacts share a 2D position after geometry "
            f"normalization (locations={positions.tolist()}). Fix "
            "Probe.Electrode rel_x/rel_y/rel_z for this sort group; the "
            "tetrode_12.5 repair applies only to 4-channel single-group "
            "tetrodes."
        )


def flatten_planar_geometry(recording) -> None:
    """Drop the constant third coordinate from a reloaded recording, in place.

    SpikeInterface reloads artifacts with a 3D ``location`` because the
    writer persists ``rel_z = 0``. No-op when locations are 2D or absent.

    Parameters
    ----------
    recording : si.BaseRecording
        The reloaded recording.

    Raises
    ------
    ValueError
        If the third coordinate is not constant across contacts: the geometry
        was never reduced to a plane, so dropping z would move contacts.
    """
    import numpy as np

    locations = recording.get_property("location")
    if locations is None or np.asarray(locations).shape[1] != 3:
        return
    locations = np.asarray(locations, dtype=float)
    if len(np.unique(np.round(locations[:, 2], _POSITION_DECIMALS))) != 1:
        raise ValueError(
            "The recording's contacts are not planar -- their third "
            f"coordinate varies (rel_z={locations[:, 2].tolist()}), so it "
            "cannot be dropped. The stored geometry was not normalized to a "
            "plane; check Probe.Electrode rel_x/rel_y/rel_z and re-populate "
            "the Recording rows."
        )
    recording.set_channel_locations(locations[:, :2])


# ``interpolate`` adjacency, in multiples of the probe's own pitch.
MIN_GOOD_NEIGHBORS = 2  # surrounded (>=2), not merely adjacent on one side
RADIUS_FACTOR = 1.5  # one pitch away counts; a multi-pitch gap does not


def _shank_pitch(shank_xyz):
    """Compute the median nearest-neighbor distance over a shank.

    ``shank_xyz``: (M, 3) ``rel_*`` positions of every electrode on the
    shank, good and bad. Returns ``None`` for fewer than 2 electrodes, any
    non-finite coordinate, or a non-positive spacing.
    """
    import numpy as np

    xyz = np.asarray(shank_xyz, dtype=float)
    if xyz.shape[0] < 2 or not np.isfinite(xyz).all():
        return None
    dd = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(dd, np.inf)
    pitch = float(np.median(dd.min(axis=1)))
    return pitch if pitch > 0 else None


def _interior_bad_channel_ids(good_xyz, candidate_xyz, pitch):
    """Select curated-bad ids embedded among a group's good channels.

    ``good_xyz``: (N, 3) ``rel_*`` positions of the good channels.
    ``candidate_xyz``: ``(electrode_id, (rel_x, rel_y, rel_z))`` per
    curated-bad electrode on the shank(s). ``pitch``: from
    :func:`_shank_pitch`. A candidate is kept when at least
    ``MIN_GOOD_NEIGHBORS`` good channels lie within ``RADIUS_FACTOR *
    pitch``. Non-finite input yields ``[]`` or skips the candidate. Returns
    a sorted list.
    """
    import numpy as np

    good = np.asarray(good_xyz, dtype=float)
    if (
        good.shape[0] < 2
        or not pitch
        or not np.isfinite(pitch)
        or not np.isfinite(good).all()
    ):
        return []  # need >=2 finite good channels and a finite pitch
    radius = RADIUS_FACTOR * float(pitch)
    return sorted(
        int(cid)
        for cid, pos in candidate_xyz
        if np.isfinite(pos).all()
        and int(
            (
                np.linalg.norm(good - np.asarray(pos, float), axis=1) <= radius
            ).sum()
        )
        >= MIN_GOOD_NEIGHBORS
    )


def fetch_interior_bad_channel_ids(
    nwb_file_name: str, sort_group_channel_ids
) -> tuple:
    """Fetch the sort group's *interior* curated-bad electrode ids.

    The curated-bad electrodes on the group's ``(electrode_group_name,
    probe_shank)`` that are embedded among its good channels (see
    :func:`_interior_bad_channel_ids`). Uses ``Probe.Electrode`` ``rel_*``,
    the same coordinates SpikeInterface interpolates with;
    ``Electrode.x/y/z`` are brain coordinates and often unset.

    Raises ``ValueError`` when a needed position or probe link is missing or
    a shank's pitch is undefined.

    Parameters
    ----------
    nwb_file_name : str
        Parent NWB filename to restrict the geometry fetch.
    sort_group_channel_ids : sequence of int
        Spyglass electrode ids of the sort group's good (member)
        channels.

    Returns
    -------
    tuple
        Sorted tuple of interior curated-bad electrode ids to
        re-include and fill on the ``interpolate`` path; empty when
        the group has no interior bad channels.
    """
    import numpy as np

    from spyglass.common.common_device import Probe
    from spyglass.common.common_ephys import Electrode

    def _fail(what: str):
        raise ValueError(
            "Recording.make: bad_channel_handling='interpolate' needs probe "
            f"geometry for {nwb_file_name!r}, but {what}. Use "
            "bad_channel_handling='remove', or populate Probe.Electrode "
            "rel_x/rel_y/rel_z and probe_shank for the sort group's shank(s)."
        )

    good_ids = {int(c) for c in sort_group_channel_ids}
    # Electrodes without a probe link drop out of the join.
    good_rows = (
        (Electrode * Probe.Electrode)
        & {"nwb_file_name": nwb_file_name}
        & [{"electrode_id": c} for c in sorted(good_ids)]
    ).fetch(
        "electrode_id",
        "electrode_group_name",
        "probe_shank",
        "rel_x",
        "rel_y",
        "rel_z",
        as_dict=True,
    )
    found_ids = {int(r["electrode_id"]) for r in good_rows}
    if found_ids != good_ids:
        missing = sorted(good_ids - found_ids)
        _fail(
            f"sort-group electrode(s) {missing[:5]} have no probe geometry "
            "(no Probe.Electrode link)"
        )

    good_by_shank: dict[tuple[str, int], list] = {}
    for r in good_rows:
        if r["probe_shank"] is None:
            _fail(f"electrode {int(r['electrode_id'])} has no probe_shank")
        pos = np.array([r["rel_x"], r["rel_y"], r["rel_z"]], dtype=float)
        if not np.isfinite(pos).all():
            _fail(
                f"good electrode {int(r['electrode_id'])} has a null position"
            )
        good_by_shank.setdefault(
            (str(r["electrode_group_name"]), int(r["probe_shank"])), []
        ).append(pos)

    interior: list[int] = []
    for (egroup, shank), good_xyz in good_by_shank.items():
        restr = {
            "nwb_file_name": nwb_file_name,
            "electrode_group_name": egroup,
            "probe_shank": shank,
        }
        shank_rows = ((Electrode * Probe.Electrode) & restr).fetch(
            "electrode_id",
            "bad_channel",
            "rel_x",
            "rel_y",
            "rel_z",
            as_dict=True,
        )
        shank_xyz = np.array(
            [[r["rel_x"], r["rel_y"], r["rel_z"]] for r in shank_rows],
            dtype=float,
        )
        pitch = _shank_pitch(shank_xyz)
        if pitch is None:
            _fail(
                f"shank {shank} of group {egroup!r} has < 2 positioned "
                "electrodes or a null coordinate (pitch undefined)"
            )
        # Bad channels already in the group need no re-inclusion.
        candidate_xyz = [
            (
                int(r["electrode_id"]),
                np.array([r["rel_x"], r["rel_y"], r["rel_z"]], dtype=float),
            )
            for r in shank_rows
            if r["bad_channel"] == "True"
            and int(r["electrode_id"]) not in good_ids
        ]
        interior.extend(
            _interior_bad_channel_ids(good_xyz, candidate_xyz, pitch)
        )
    return tuple(sorted(set(interior)))
