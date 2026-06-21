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
dependencies are imported lazily inside the functions. Three functions
inherently touch the DB / DataJoint at CALL time via lazy imports:
``spikeinterface_channel_ids`` (an ``Nwbfile`` path resolution),
``fetch_sort_group_probe_info`` (an ``Electrode * Probe`` fetch), and
``fetch_interior_bad_channel_ids`` (an ``Electrode * Probe.Electrode`` fetch).
``maybe_apply_tetrode_geometry`` and the pitch/adjacency helpers
(``_shank_pitch``, ``_interior_bad_channel_ids``) are pure.
"""

from __future__ import annotations


def spikeinterface_channel_ids(nwb_file_name: str, spyglass_ids):
    """Map Spyglass electrode_ids onto SpikeInterface channel ids.

    SpikeInterface 0.104's ``read_nwb_recording`` uses the raw NWB
    electrodes table's ``channel_name`` string column as the
    channel id if present; otherwise integer ``electrode_id`` is
    the channel id (the 1-1 fallback). Production NWBs that carry a
    ``channel_name`` column resolve correctly through it. The Frank-lab
    MEArec fixture lacks the column and falls through to the
    integer path.

    Parameters
    ----------
    nwb_file_name : str
        Parent NWB filename whose electrodes table is read.
    spyglass_ids : sequence of int
        Spyglass electrode ids to map onto SI channel ids.

    Returns
    -------
    list
        SpikeInterface channel ids -- ``channel_name`` strings when
        the electrodes table carries that column, otherwise integer
        electrode ids.
    """
    import pynwb

    from spyglass.common.common_nwbfile import Nwbfile

    nwb_file_abs_path = Nwbfile.get_abs_path(nwb_file_name)
    with pynwb.NWBHDF5IO(nwb_file_abs_path, mode="r") as io:
        nwbfile = io.read()
        electrodes_table = nwbfile.electrodes
        if "channel_name" not in electrodes_table.colnames:
            return [int(c) for c in spyglass_ids]
        channel_names = electrodes_table["channel_name"]
        return [channel_names[int(c)] for c in spyglass_ids]


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

    unique_probes = set(probe_types)
    unique_groups = set(electrode_group_names)
    # First failing gate wins; the reason text lives next to its
    # predicate so adding/removing a gate is a one-line edit with no
    # index alignment. ``next(iter(...), None)`` avoids StopIteration on
    # an empty probe set -- the ``len != 1`` gate above it fires first.
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
            len(sort_group_channel_ids) != 4,
            "sort group does not have exactly 4 channels",
        ),
        (
            len(unique_groups) != 1,
            "sort group spans multiple electrode groups",
        ),
    )
    for failed, reason in gates:
        if failed:
            logger.info("_maybe_apply_tetrode_geometry skipped: %s", reason)
            return recording

    import numpy as _np
    import probeinterface as pi

    tetrode = pi.Probe(ndim=2)
    position = [[0, 0], [0, 12.5], [12.5, 0], [12.5, 12.5]]
    tetrode.set_contacts(
        position,
        shapes="circle",
        shape_params={"radius": 6.25},
    )
    tetrode.set_contact_ids([str(c) for c in sort_group_channel_ids])
    tetrode.set_device_channel_indices(_np.arange(4))
    return recording.set_probe(tetrode, in_place=True)


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
        # sort-group members (a remove_bad_channels=False group keeps its own
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
