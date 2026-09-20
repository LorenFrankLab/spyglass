"""Channel-id / probe-geometry services behind ``Recording``.

These functions resolve SpikeInterface channel ids and per-channel probe
metadata for ``Recording.make_compute`` (and the rebuild path), attach the
legacy ``tetrode_12.5`` geometry to four-channel tetrode groups, and select the
interior curated-bad channels the ``interpolate`` path re-includes. The table
threads already-fetched DB state in (the tri-part
``make_fetch``/``make_compute``/``make_insert`` contract forbids DB I/O inside
compute), so the geometry math here is DB-free.

Why this lives in its own module rather than in ``recording.py``:
``recording.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part dependencies. The geometry logic needs
none of that at import, so ``Recording`` becomes a thin orchestrator (fetch ->
call these -> insert / verify). Same "thin DataJoint shell over pure/IO
services" direction as ``_artifact_compute`` / ``_selection_identity`` /
``_analyzer_cache`` / ``_curation_transforms`` / ``_units_nwb`` /
``_sorting_dispatch``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: all SpikeInterface / numpy / probeinterface / spyglass
dependencies are imported lazily inside the functions. Four functions
inherently touch the DB / DataJoint at CALL time via lazy imports:
``spikeinterface_channel_ids`` (an ``Nwbfile`` path resolution),
``fetch_sort_group_probe_info`` (an ``Electrode * Probe`` fetch),
``fetch_sort_group_contact_positions`` and ``fetch_interior_bad_channel_ids``
(both an ``Electrode * Probe.Electrode`` fetch).
``maybe_apply_tetrode_geometry``, its gate predicate
(``tetrode_repair_applies``), the plane-normalization helpers
(``select_distinct_plane``, ``normalize_channel_locations``,
``assert_unique_contact_positions``) and the pitch/adjacency helpers
(``_shank_pitch``, ``_interior_bad_channel_ids``) are pure.
"""

from __future__ import annotations


def spikeinterface_channel_ids(nwb_file_name: str, spyglass_ids):
    """Map Spyglass electrode_ids onto SpikeInterface channel ids.

    SpikeInterface 0.104's ``read_nwb_recording`` uses the raw NWB
    electrodes table's ``channel_name`` string column as the channel id if
    present; otherwise the integer ``electrode_id`` is the channel id (the 1-1
    fallback). The ``channel_name`` column is POSITIONAL -- row ``k`` holds
    channel ``k``'s name -- but Spyglass electrode ids are not guaranteed to
    equal table row positions (production NWBs can be non-contiguous,
    non-zero-based, or reordered). Each electrode id is therefore resolved to
    its electrodes-table ROW via ``get_electrode_indices`` (the same id->row
    mapping the write side uses in ``electrode_table_region``) before reading
    ``channel_name`` -- never ``channel_name[electrode_id]``, which would slice
    the wrong channel. A requested id that is absent, or that maps to more than
    one row, raises rather than silently mis-indexing.

    Parameters
    ----------
    nwb_file_name : str
        Parent NWB filename whose electrodes table is read.
    spyglass_ids : sequence of int
        Spyglass electrode ids to map onto SI channel ids.

    Returns
    -------
    list
        SpikeInterface channel ids -- ``channel_name`` strings (in the order
        of ``spyglass_ids``) when the electrodes table carries that column,
        otherwise the integer electrode ids.

    Raises
    ------
    ValueError
        If any requested electrode id is not in the electrodes table, or maps
        to more than one row (ambiguous id).
    """
    import pynwb

    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.utils.nwb_helper_fn import (
        get_electrode_indices,
        invalid_electrode_index,
    )

    ids = [int(c) for c in spyglass_ids]
    nwb_file_abs_path = Nwbfile.get_abs_path(nwb_file_name)
    with pynwb.NWBHDF5IO(nwb_file_abs_path, mode="r") as io:
        nwbfile = io.read()
        electrodes_table = nwbfile.electrodes
        table_ids = [int(e) for e in electrodes_table.id[:]]
        # Map each requested electrode id -> electrodes-table row index.
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
            # No channel_name column: SI uses the integer electrode_id as the
            # channel id (validated present above), independent of row order.
            return ids
        channel_names = electrodes_table["channel_name"]
        return [channel_names[idx] for idx in row_indices]


def fetch_sort_group_probe_info(
    nwb_file_name: str, channel_ids
) -> tuple[tuple, tuple]:
    """Fetch per-channel ``probe_type`` + ``electrode_group_name``.

    Returns a pair of tuples (probe_types, electrode_group_names),
    one entry per channel id in ``channel_ids``. The tuple form
    is DeepHash-stable (NamedTuple field constraint for
    ``RecordingFetched``). Used by both the populate path
    (``make_fetch``) and the rebuild path
    (``_rebuild_nwb_artifact``) to feed
    ``maybe_apply_tetrode_geometry``.

    The fetch is ``order_by="electrode_id"`` so two successive
    ``make_fetch`` calls return byte-identical tuples; without an
    explicit ordering, DataJoint/MySQL row order is unspecified and
    the tri-part DeepHash integrity check inside the populate
    transaction can spuriously raise on reorder. This matches the
    ordered-fetch pattern used by the other tri-part ``make_fetch``
    paths in this package (e.g. ``ArtifactDetection.make_fetch`` in
    artifact.py, which orders its member fetch by ``recording_id``).

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

    The probe-relative ``rel_x``/``rel_y``/``rel_z`` columns (joined onto
    ``Electrode``), NOT ``Electrode.x/y/z`` -- the same source
    :func:`fetch_interior_bad_channel_ids` uses.

    **This is the registered probe's geometry, not the session's.**
    ``Probe.Electrode`` is ingested from the NWB's ``ShanksElectrode`` device
    metadata and keyed by probe TYPE, so one row set serves every session
    recorded on that probe type. SpikeInterface, by contrast, reads the
    per-session electrodes table's ``rel_x``/``rel_y``/``rel_z`` columns. The
    two normally agree, because Frank-lab writers fill both from the same
    probe YAML -- but nothing enforces it, and a file whose electrodes table
    was written (or edited) independently of its device metadata will
    disagree. The v2 suite has such a fixture on purpose: a synthesized
    tetrode whose electrodes table is zeroed while its ``ShanksElectrode``
    attributes keep the canonical square. Callers reasoning about what the
    SORT will see should keep that gap in mind.

    The columns are nullable and the join drops any electrode with no probe
    link, so both "no geometry" cases arrive as ``NaN`` rows rather than a
    silent 0.0: the caller decides whether that means "this group has no
    geometry at all" (the legacy all-zero tetrode, which
    :func:`maybe_apply_tetrode_geometry` repairs) or a partially-populated
    probe (an error).

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
        row-aligned to ``sorted(channel_ids)`` -- the order
        :func:`fetch_sort_group_probe_info` returns its tuples in, so the two
        can be zipped. ``NaN`` where the electrode has no probe link or a
        NULL coordinate.
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


# Gates for the legacy ``tetrode_12.5`` geometry repair, as
# (predicate-of-failure, reason) pairs built from the sort group's probe
# metadata. The reason text lives next to its predicate so adding/removing a
# gate is a one-line edit with no index alignment.
def _tetrode_repair_skip_reason(
    probe_types: tuple, electrode_group_names: tuple, n_channels: int
) -> "str | None":
    """Why the ``tetrode_12.5`` repair does NOT apply, or ``None`` if it does.

    Separated from :func:`maybe_apply_tetrode_geometry` so preflight can ask
    the same question without a recording (see
    :func:`tetrode_repair_applies`), and so the log message an operator greps
    for stays next to the predicate that produced it.
    """
    unique_probes = set(probe_types)
    unique_groups = set(electrode_group_names)
    # ``next(iter(...), None)`` avoids StopIteration on an empty probe set --
    # the ``len != 1`` gate above it fires first.
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

    Preflight uses this to decide whether a sort group whose stored geometry
    collapses in every plane (the legacy all-zero tetrode) is nevertheless
    runnable: the repair installs the 12.5 µm square before the uniqueness
    assertion runs, so those groups must not be reported as broken.

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

    Sort groups of exactly 4 channels on a single ``tetrode_12.5``
    probe and a single electrode group get an explicit
    ``(0,0)-(0,12.5)-(12.5,0)-(12.5,12.5)`` µm probe with 6.25 µm
    contact radius. Covers legacy Frank-lab NWBs where contact
    positions were never written into the electrode table.
    Geometry-aware sorters (Kilosort, MountainSort5) on those
    recordings depend on this patch; clusterless_thresholder and
    MS4 are unaffected.

    When any gate fails the recording is returned untouched and an
    ``INFO`` log names the failed gate's reason, so an operator
    debugging "Kilosort sees the wrong geometry" can grep the populate
    log for which condition skipped the patch.

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

    # First failing gate wins. The gates themselves live in
    # ``_tetrode_repair_skip_reason`` so preflight (via
    # ``tetrode_repair_applies``) cannot drift from what this actually does.
    reason = _tetrode_repair_skip_reason(
        probe_types, electrode_group_names, len(sort_group_channel_ids)
    )
    if reason is not None:
        logger.info("maybe_apply_tetrode_geometry skipped: %s", reason)
        return recording

    import numpy as np
    import probeinterface as pi

    tetrode = pi.Probe(ndim=2)
    position = [[0, 0], [0, 12.5], [12.5, 0], [12.5, 12.5]]
    tetrode.set_contacts(
        position,
        shapes="circle",
        shape_params={"radius": 6.25},
    )
    tetrode.set_contact_ids([str(c) for c in sort_group_channel_ids])
    tetrode.set_device_channel_indices(np.arange(4))
    return recording.set_probe(tetrode, in_place=True)


# Axis pairs tried, in order, when reducing 3D contact positions to the plane
# SpikeInterface will actually use. ``x-y`` first because that is what
# probeinterface's ``select_axes`` default gives, so a genuinely planar x-y
# probe is never re-projected.
_CANDIDATE_PLANES = ("xy", "xz", "yz")

# Coordinates are micrometres read from ``Probe.Electrode`` ``rel_*`` columns
# (``float``, i.e. single precision in DataJoint); 6 decimals is far below the
# smallest real contact pitch and above float32 representation noise, so
# rounding here decides "same contact" without float equality surprises.
_POSITION_DECIMALS = 6


def _all_rows_distinct(positions) -> bool:
    """True when no two rows of ``positions`` coincide (to 6 decimals)."""
    import numpy as np

    rounded = np.round(np.asarray(positions, dtype=float), _POSITION_DECIMALS)
    return len(np.unique(rounded, axis=0)) == len(rounded)


def select_distinct_plane(locations):
    """Pick the axis pair in which every contact has a distinct position.

    SpikeInterface reduces 3D channel locations to 2D whenever it builds a
    probe -- ``create_dummy_probe_from_locations`` calls
    ``probeinterface.select_axes(locations, axes)`` with ``axes="xy"`` -- and
    a probe with two contacts at the same position is rejected. Real
    Frank-lab tetrodes lie in the x-z plane (``rel_y`` constant), so their x-y
    projection collapses pairs of contacts; this picks x-z instead.

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
        ``None`` when no pair separates every contact -- including the
        all-zero legacy geometry, where no projection can.

    Raises
    ------
    ValueError
        If ``locations`` is not ``(n_contacts, 3)``, or any coordinate is
        non-finite.
    """
    import numpy as np

    loc = np.asarray(locations, dtype=float)
    if loc.ndim != 2 or loc.shape[1] != 3:
        raise ValueError(
            "select_distinct_plane: expected an (n_contacts, 3) array of "
            f"contact positions, got shape {loc.shape}"
        )
    # ``rel_x/rel_y/rel_z`` are nullable, so a NULL arrives as NaN. NaN rows
    # compare as distinct under np.unique, which would let a NaN plane win and
    # sail through assert_unique_contact_positions into a probe build. Same
    # finiteness screen ``_shank_pitch`` applies below.
    if not np.isfinite(loc).all():
        bad = np.flatnonzero(~np.isfinite(loc).all(axis=1))
        raise ValueError(
            "select_distinct_plane: contact positions must be finite, but "
            f"row(s) {bad.tolist()} are {loc[bad].tolist()}. Populate "
            "Probe.Electrode rel_x/rel_y/rel_z for this sort group."
        )
    for axes in _CANDIDATE_PLANES:
        positions = loc[:, ["xyz".index(axis) for axis in axes]]
        if _all_rows_distinct(positions):
            return axes, positions
    return None


def normalize_channel_locations(recording, *, channel_ids=None):
    """Reduce 3D channel locations to the distinct 2D plane, in place.

    Runs at the recording stage on the channel-sliced recording -- so a group
    whose bad channels were removed is normalized on the channels it actually
    contains -- and *before* any probe exists, because SpikeInterface's
    ``set_channel_locations`` refuses to write locations once a
    ``contact_vector`` is attached (it would silently desynchronize the probe
    from the property). Nothing here builds a probe.

    A recording whose locations are already 2D, or that carries no ``location``
    property at all, is returned untouched. So is one no plane separates: the
    legacy all-zero geometry is repaired downstream by
    :func:`maybe_apply_tetrode_geometry`, and
    :func:`assert_unique_contact_positions` raises afterwards if it was not.
    A set in which EVERY coordinate is non-finite is treated the same way --
    as no usable geometry rather than as an error -- because that is exactly
    what ``sort_group_geometry_problem`` clears at preflight (it maps such a
    group onto the all-zero legacy geometry so the ``tetrode_12.5`` repair can
    rescue it). Raising here would fail at make a group preflight passed.
    PARTIAL non-finite rows still raise: the repair does not apply to them, so
    the missing contacts are a defect no downstream step can fix.

    SUBSET RULE. The plane is chosen from the rows of ``channel_ids`` -- the
    contacts that REMAIN on the sort surface -- and the chosen projection is
    then applied to EVERY channel of the recording. The channel slice is wider
    than that surface: it also carries the ``specific`` reference electrode,
    which is subtracted and dropped in ``apply_spatial_preprocessing``.
    ``Probe.Electrode`` ``rel_*`` are recorded per probe TYPE, so a reference
    on another probe of the same type has the same raw coordinates as one of
    the members; letting it take part in the choice means no plane separates
    the sliced set, the recording stays 3D, and
    :func:`assert_unique_contact_positions` then rejects member geometry that
    is perfectly distinct on its own. The reference still gets 2D coordinates
    because preprocessing and the writer require one consistent location
    array across the surface -- it is dropped before the uniqueness
    assertion, so its (possibly duplicated) position never has to be distinct.

    Parameters
    ----------
    recording : si.BaseRecording
        The channel-sliced recording, with no probe attached.
    channel_ids : sequence, optional
        Keyword-only. Channel ids of the contacts the plane is chosen from;
        ``None`` (default) uses every channel. Pass the contacts the sort
        surface retains: the members plus, on the ``interpolate`` path, the
        interior bad channels -- everything except the ``specific``
        reference.

    Returns
    -------
    si.BaseRecording
        The same recording, with 2D channel locations when a plane was chosen.

    Raises
    ------
    ValueError
        If a probe is already attached to ``recording``, if ``channel_ids``
        names a channel the recording does not carry, or if SOME but not all
        of the retained 3D contact positions are non-finite (a NULL ``rel_*``
        column).
    """
    import numpy as np

    from spyglass.utils import logger

    if recording.get_property("contact_vector") is not None:
        raise ValueError(
            "normalize_channel_locations: a probe is already attached to this "
            "recording, so its channel locations can no longer be rewritten. "
            "Normalize the geometry before any probe is built (before "
            "set_probe / get_probe / create_sorting_analyzer)."
        )
    # get_channel_locations() raises when there is no location property, and
    # axes="xyz" on a 2D property indexes out of bounds, so the property
    # itself is the ndim test.
    locations = recording.get_property("location")
    if locations is None or np.asarray(locations).shape[1] != 3:
        return recording

    positions_3d = np.asarray(
        recording.get_channel_locations(axes="xyz"), dtype=float
    )
    # ``ids_to_indices`` raises naming the unknown ids, which is the report an
    # operator needs when the retained set and the slice have drifted apart.
    retained = (
        positions_3d
        if channel_ids is None
        else positions_3d[recording.ids_to_indices(list(channel_ids))]
    )
    if not np.isfinite(retained).any():
        # No coordinate at all is indistinguishable from an absent location
        # property, and preflight clears exactly this case so the tetrode
        # repair can run; ``select_distinct_plane`` would raise on it.
        return recording

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
    # The projection the retained contacts chose, applied to every channel.
    recording.set_channel_locations(
        positions_3d[:, ["xyz".index(axis) for axis in axes]]
    )
    return recording


def assert_unique_contact_positions(
    recording, *, require_2d: bool = True
) -> None:
    """Require every contact to have a distinct 2D position.

    Reads ``get_channel_locations()`` -- the x-y projection of whatever is
    attached, probe or bare property -- because that is the geometry
    SpikeInterface will hand to ``create_sorting_analyzer``. Run this last, so
    the *effective* geometry is what is checked. Single-channel groups pass
    trivially.

    With ``require_2d`` (the default) a recording whose ``location`` property
    is STILL 3D is refused here rather than deep inside the artifact writer.
    The recording stage can reach that state: a ``specific`` reference channel
    takes part in plane selection and is dropped afterwards, so a group whose
    contacts separate in no plane WITH the reference can end up 3D but
    x-y-distinct without it -- which the coincidence check below waves through
    and ``write_nwb_artifact`` then rejects, after the whole compute has run.

    Parameters
    ----------
    recording : si.BaseRecording
        The fully prepared recording, after normalization and any probe patch.
    require_2d : bool, default True
        Keyword-only. When False, 3D locations are projected to x-y for the
        uniqueness check instead of being refused. Consumers that reload a
        written artifact pass False: the writer persists ``rel_z``, so
        ``NwbRecordingExtractor`` rebuilds a 3D ``location`` property for every
        artifact, and those callers project it deliberately.

    Raises
    ------
    ValueError
        If the recording carries no geometry at all, if ``require_2d`` and its
        locations are still 3D, or if two or more contacts share a 2D position.
    """
    import numpy as np

    # get_channel_locations raises a bare Exception("There are no channel
    # locations") when neither a probe nor a location property is present;
    # answer that case here so the operator gets the same actionable message.
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


# Pitch-anchored adjacency for the ``interpolate`` re-inclusion. Constants are
# dimensionless multiples of the probe's own physical pitch, so one rule fits
# dense Neuropixels shanks and sparse polymer groups alike.
MIN_GOOD_NEIGHBORS = 2  # surrounded (>=2), not merely adjacent on one side
RADIUS_FACTOR = 1.5  # one pitch away counts; a multi-pitch gap does not


def _shank_pitch(shank_xyz):
    """Compute the median nearest-neighbor distance over a shank.

    ``shank_xyz``: (M, 3) probe-relative positions (``Probe.Electrode``
    ``rel_x/rel_y/rel_z``) of every electrode on the shank (good AND bad), so
    the result is the probe's physical pitch, independent of which channels a
    sort group happens to keep. Returns ``None`` when the shank has < 2
    electrodes, any coordinate is non-finite (``rel_x/rel_y/rel_z`` are
    nullable -> a NULL arrives as NaN), OR the spacing is non-positive
    (coincident / duplicate contact positions give a 0 median). A ``None``
    pitch makes the caller raise the clear "needs positions" error rather than
    silently producing NaN distances or a 0 pitch that re-includes nothing.
    """
    import numpy as np

    xyz = np.asarray(shank_xyz, dtype=float)
    if xyz.shape[0] < 2 or not np.isfinite(xyz).all():
        return None
    dd = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(dd, np.inf)
    pitch = float(np.median(dd.min(axis=1)))
    # A non-positive median (coincident/duplicate contacts) is degenerate
    # geometry the adjacency test can't use; treat it as undefined so the caller
    # raises rather than the falsy-0 path silently returning an empty set.
    return pitch if pitch > 0 else None


def _interior_bad_channel_ids(good_xyz, candidate_xyz, pitch):
    """Select curated-bad ids embedded among a group's good channels.

    ``good_xyz``: (N, 3) probe-relative positions of the group's good channels.
    ``candidate_xyz``: list of ``(electrode_id, rel_position_array)`` for the
    curated-bad electrodes on the group's shank(s) (the position is a
    ``(rel_x, rel_y, rel_z)`` array or tuple). ``pitch``: the shank's
    physical electrode spacing from :func:`_shank_pitch` (NOT derived from the
    possibly-sparse good set). A candidate is kept only when at least
    ``MIN_GOOD_NEIGHBORS`` good channels lie within ``RADIUS_FACTOR * pitch`` of
    it -- so a bad channel between two far-apart good channels (its nearest good
    channel many pitches away) is excluded, while a bad channel in a dense run
    is kept. Returns a sorted list. Defensive on non-finite input
    (``make_fetch`` raises before calling): a non-finite ``pitch`` or good
    position -> ``[]``; a candidate with a non-finite position is skipped (never
    silently treated as adjacent).
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

    For the ``interpolate`` bad-channel-handling path.

    Returns a sorted tuple of the curated-bad (``Electrode.bad_channel='True'``)
    electrodes on the sort group's shank(s) that are physically embedded among
    the group's good channels (>= ``MIN_GOOD_NEIGHBORS`` good channels within
    ``RADIUS_FACTOR * pitch``, ``pitch`` the full-shank spacing). These are the
    channels ``interpolate`` re-includes and fills. The candidate set is scoped
    to the group's ``(electrode_group_name, probe_shank)`` -- NOT a shank-wide
    grab -- because ``set_group_by_electrode_table_column`` builds
    arbitrary-membership groups and stores no original column values.

    Geometry comes from the probe-relative ``Probe.Electrode``
    ``rel_x/rel_y/rel_z`` (joined onto ``Electrode``), NOT ``Electrode.x/y/z``:
    the latter are absolute brain coordinates that are commonly unset (all
    zero / NULL), whereas ``rel_*`` is the physical contact geometry -- the
    same coordinate system SpikeInterface's ``get_channel_locations`` (and
    therefore ``interpolate_bad_channels``) reads from the NWB. Using the same
    source keeps the adjacency decision consistent with the actual fill.

    Raises ``ValueError`` (pointing the user at ``remove``) when any required
    position is null/NaN, a needed electrode has no probe geometry, or a shank's
    pitch is undefined -- the full-shank position fetch is a superset of the
    good + candidate positions, so a ``_shank_pitch`` of ``None`` is the
    finiteness gate for the whole shank (rather than silently returning an empty
    set that masquerades as "no bad channels to fill"). DB-touching at call time
    via lazy ``Electrode`` / ``Probe`` imports; the geometry math is delegated to
    the pure helpers above.

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
    # Inner-join Electrode with Probe.Electrode to bring in rel_x/rel_y/rel_z
    # (and probe_shank). Electrodes lacking a probe link drop out of the join,
    # so an incomplete result means a member without geometry -> fail loud.
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

    # Group the good channels by physical shank.
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
        # Full-shank positions -> physical pitch (and the finiteness gate).
        shank_xyz = np.array(
            [
                [r["rel_x"], r["rel_y"], r["rel_z"]]
                for r in ((Electrode * Probe.Electrode) & restr).fetch(
                    "rel_x", "rel_y", "rel_z", as_dict=True
                )
            ],
            dtype=float,
        )
        pitch = _shank_pitch(shank_xyz)
        if pitch is None:
            _fail(
                f"shank {shank} of group {egroup!r} has < 2 positioned "
                "electrodes or a null coordinate (pitch undefined)"
            )
        # Curated-bad candidates on this shank, excluding any that are already
        # sort-group members (an omit_bad_channels=False group keeps its own
        # bad members present; only the *excluded* interior bad are re-included).
        candidate_xyz = [
            (
                int(r["electrode_id"]),
                np.array([r["rel_x"], r["rel_y"], r["rel_z"]], dtype=float),
            )
            for r in (
                (Electrode * Probe.Electrode) & restr & {"bad_channel": "True"}
            ).fetch("electrode_id", "rel_x", "rel_y", "rel_z", as_dict=True)
            if int(r["electrode_id"]) not in good_ids
        ]
        interior.extend(
            _interior_bad_channel_ids(good_xyz, candidate_xyz, pitch)
        )
    return tuple(sorted(set(interior)))
