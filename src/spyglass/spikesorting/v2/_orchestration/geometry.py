"""Sort-group geometry reporting and plotting.

``pipeline.py`` re-exports ``describe_sort_groups`` and
``plot_sort_group_geometry``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spyglass.spikesorting.v2._recording.unit_metadata import (
    sort_group_electrode_regions,
)

if TYPE_CHECKING:
    import pandas as pd


# Auto-label electrode ids only for groups this sparse (tetrodes, stereotrodes).
_AUTO_LABEL_MAX_PER_COLUMN = 8


_SORT_GROUP_COLUMNS = [
    "nwb_file_name",
    "sort_group_id",
    "n_electrodes",
    "electrode_ids",
    "electrode_group_names",
    "probe_shanks",
    "brain_regions",
    "bad_channel_count",
    "reference_mode",
    "reference_electrode_id",
]


def _missing(value) -> bool:
    """Return True when ``value`` is ``None`` or a pandas/NumPy NA/NaN."""
    import pandas as pd

    return value is None or bool(pd.isna(value))


def _nullable_int(value):
    """Coerce a possibly-missing numeric to ``int``, or ``None`` if missing."""
    return None if _missing(value) else int(value)


def _probe_key(electrode: dict) -> "tuple[Any, int, int] | None":
    """Return an electrode's ``Probe.Electrode`` key, or ``None`` if incomplete.

    Parameters
    ----------
    electrode : dict
        An ``Electrode`` row (sort-group member or reference electrode).

    Returns
    -------
    tuple or None
        ``(probe_id, probe_shank, probe_electrode)`` with the shank and
        electrode as ``int``, or ``None`` when any of the three is missing.
    """
    probe_id = electrode.get("probe_id")
    probe_shank = electrode.get("probe_shank")
    probe_electrode = electrode.get("probe_electrode")
    if any(
        _missing(value) for value in (probe_id, probe_shank, probe_electrode)
    ):
        return None
    return probe_id, int(probe_shank), int(probe_electrode)


def describe_sort_groups(nwb_file_name: str) -> "pd.DataFrame":
    """Return a notebook-friendly summary of v2 sort groups for a session.

    Use before choosing a ``sort_group_id`` for ``run_v2_pipeline``.
    Read-only.

    Parameters
    ----------
    nwb_file_name : str
        Session whose existing ``SortGroupV2`` rows should be summarized.

    Returns
    -------
    pandas.DataFrame
        One row per sort group, sorted by ``sort_group_id``. Empty, with
        the documented columns, when the session has no sort groups.
        Columns are ``nwb_file_name``, ``sort_group_id``, ``n_electrodes``,
        ``electrode_ids``, ``electrode_group_names``, ``probe_shanks``,
        ``brain_regions``, ``bad_channel_count``, ``reference_mode``, and
        ``reference_electrode_id``.
    """
    import pandas as pd

    from spyglass.spikesorting.v2.recording import SortGroupV2

    def _sorted_nullable_ints(values):
        normalized = {_nullable_int(value) for value in values}
        return tuple(
            sorted(
                normalized,
                key=lambda value: (
                    value is None,
                    value if value is not None else 0,
                ),
            )
        )

    master_rows = (SortGroupV2 & {"nwb_file_name": nwb_file_name}).fetch(
        as_dict=True
    )
    if not master_rows:
        return pd.DataFrame(columns=_SORT_GROUP_COLUMNS)

    rows_by_group: dict[int, list[dict]] = {}
    for row in sort_group_electrode_regions(
        {"nwb_file_name": nwb_file_name}
    ).fetch(as_dict=True):
        rows_by_group.setdefault(int(row["sort_group_id"]), []).append(row)

    rows = []
    for master in sorted(
        master_rows, key=lambda row: int(row["sort_group_id"])
    ):
        sort_group_id = int(master["sort_group_id"])
        electrode_rows = rows_by_group.get(sort_group_id, [])

        reference_mode = master["reference_mode"]
        reference_electrode_id = (
            _nullable_int(master["reference_electrode_id"])
            if reference_mode == "specific"
            else None
        )
        rows.append(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
                "n_electrodes": len(electrode_rows),
                "electrode_ids": tuple(
                    sorted(int(row["electrode_id"]) for row in electrode_rows)
                ),
                "electrode_group_names": tuple(
                    sorted(
                        {
                            str(row["electrode_group_name"])
                            for row in electrode_rows
                        }
                    )
                ),
                "probe_shanks": _sorted_nullable_ints(
                    row.get("probe_shank") for row in electrode_rows
                ),
                "brain_regions": tuple(
                    sorted({str(row["region_name"]) for row in electrode_rows})
                ),
                "bad_channel_count": sum(
                    str(row.get("bad_channel")) == "True"
                    for row in electrode_rows
                ),
                "reference_mode": reference_mode,
                "reference_electrode_id": reference_electrode_id,
            }
        )
    return pd.DataFrame(rows, columns=_SORT_GROUP_COLUMNS)


def _registered_position(probe_geometry, electrode):
    """``(rel_x, rel_y, rel_z)`` of an electrode's probe contact, NaN if none."""
    import numpy as np

    geometry = probe_geometry.get(_probe_key(electrode))
    if not geometry:
        return np.full(3, np.nan)
    return np.array(
        [
            np.nan if _missing(geometry[axis]) else float(geometry[axis])
            for axis in ("rel_x", "rel_y", "rel_z")
        ]
    )


def _planar_coordinate_source(planar_source: str) -> str:
    """``"tetrode_12.5"`` or ``"probe_<axes>"`` for a planar row."""
    if planar_source == "tetrode_12.5":
        return planar_source
    return f"probe_{planar_source}"


def _sort_group_geometry_rows(nwb_file_name: str) -> list[dict[str, Any]]:
    """Return DB-backed electrode geometry rows for sort-group plotting.

    ``plot_x``/``plot_y`` are the positions the recording stage gives each
    contact (see ``sort_group_planar_positions``), read from the registered
    ``Probe.Electrode`` geometry as preflight does.
    """
    import numpy as np
    from probeinterface import select_axes

    from spyglass.common.common_device import Probe
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2._recording.geometry import (
        fetch_sort_group_probe_info,
        sort_group_planar_positions,
    )
    from spyglass.common.common_region import BrainRegion
    from spyglass.spikesorting.v2.recording import SortGroupV2

    master_rows = (SortGroupV2 & {"nwb_file_name": nwb_file_name}).fetch(
        as_dict=True
    )
    master_by_group = {
        int(row["sort_group_id"]): row
        for row in sorted(
            master_rows, key=lambda row: int(row["sort_group_id"])
        )
    }
    if not master_by_group:
        return []

    member_rows = sort_group_electrode_regions(
        {"nwb_file_name": nwb_file_name}
    ).fetch(as_dict=True)
    member_rows_by_group: dict[int, list[dict]] = {}
    for row in member_rows:
        member_rows_by_group.setdefault(int(row["sort_group_id"]), []).append(
            row
        )

    # Specific reference electrodes are not group members; fetch them directly.
    reference_ids = {
        _nullable_int(master["reference_electrode_id"])
        for master in master_by_group.values()
        if master["reference_mode"] == "specific"
        and not _missing(master["reference_electrode_id"])
    }
    reference_electrode_by_id: dict[int, dict] = {}
    if reference_ids:
        reference_electrode_by_id = {
            int(row["electrode_id"]): row
            for row in (
                (Electrode & {"nwb_file_name": nwb_file_name}) * BrainRegion
                & [{"electrode_id": rid} for rid in reference_ids]
            ).fetch(as_dict=True)
        }

    probe_restrictions = [
        dict(zip(("probe_id", "probe_shank", "probe_electrode"), probe_key))
        for electrode in [*member_rows, *reference_electrode_by_id.values()]
        if (probe_key := _probe_key(electrode)) is not None
    ]
    probe_geometry = {}
    if probe_restrictions:
        probe_rows = (Probe.Electrode & probe_restrictions).fetch(
            "probe_id",
            "probe_shank",
            "probe_electrode",
            "rel_x",
            "rel_y",
            "rel_z",
            "contact_size",
            as_dict=True,
        )
        probe_geometry = {
            (
                row["probe_id"],
                int(row["probe_shank"]),
                int(row["probe_electrode"]),
            ): row
            for row in probe_rows
        }

    rows: list[dict[str, Any]] = []
    for sort_group_id, master in master_by_group.items():
        sort_group_id = int(master["sort_group_id"])
        reference_mode = master["reference_mode"]
        reference_electrode_id = (
            _nullable_int(master["reference_electrode_id"])
            if reference_mode == "specific"
            else None
        )
        group_electrodes = sorted(
            member_rows_by_group.get(sort_group_id, []),
            key=lambda row: int(row["electrode_id"]),
        )
        # Plot contacts where the recording stage places them.
        member_ids = [int(row["electrode_id"]) for row in group_electrodes]
        planar_source, planar = (
            sort_group_planar_positions(
                np.array(
                    [
                        _registered_position(probe_geometry, row)
                        for row in group_electrodes
                    ]
                ),
                *fetch_sort_group_probe_info(nwb_file_name, member_ids),
            )
            if group_electrodes
            else (None, None)
        )
        planar_by_id = (
            dict(zip(member_ids, map(tuple, planar)))
            if planar is not None
            else {}
        )
        # The appended reference row renders as the star overlay only.
        if reference_electrode_id in reference_electrode_by_id:
            group_electrodes = [
                *group_electrodes,
                reference_electrode_by_id[reference_electrode_id],
            ]
        for electrode in group_electrodes:
            rel_x = rel_y = rel_z = contact_size = None
            probe_key = _probe_key(electrode)
            if probe_key is not None:
                geometry = probe_geometry.get(probe_key)
                if geometry:
                    rel_x = geometry["rel_x"]
                    rel_y = geometry["rel_y"]
                    rel_z = geometry["rel_z"]
                    contact_size = geometry["contact_size"]

            # Take plot_x, plot_y and coordinate_source from one source.
            electrode_x = electrode.get("x")
            electrode_y = electrode.get("y")
            electrode_id = int(electrode["electrode_id"])
            if electrode_id in planar_by_id:
                plot_x, plot_y = map(float, planar_by_id[electrode_id])
                coord_source = _planar_coordinate_source(planar_source)
            elif planar_source == "tetrode_12.5":
                # A reference off this tetrode has no place on its square.
                plot_x, plot_y, coord_source = None, None, None
            elif planar_source is not None:
                registered = _registered_position(probe_geometry, electrode)
                if np.isfinite(registered).all():
                    plot_x, plot_y = map(
                        float,
                        select_axes(registered[None, :], planar_source)[0],
                    )
                    coord_source = _planar_coordinate_source(planar_source)
                else:
                    plot_x, plot_y, coord_source = None, None, None
            elif not _missing(rel_x) and not _missing(rel_y):
                plot_x, plot_y, coord_source = rel_x, rel_y, "probe"
            elif not _missing(electrode_x) and not _missing(electrode_y):
                plot_x, plot_y, coord_source = (
                    electrode_x,
                    electrode_y,
                    "electrode",
                )
            else:
                plot_x, plot_y, coord_source = None, None, None
            rows.append(
                {
                    "nwb_file_name": nwb_file_name,
                    "sort_group_id": sort_group_id,
                    "electrode_id": int(electrode["electrode_id"]),
                    "electrode_group_name": str(
                        electrode["electrode_group_name"]
                    ),
                    "probe_id": electrode.get("probe_id"),
                    "probe_shank": _nullable_int(electrode.get("probe_shank")),
                    "probe_electrode": _nullable_int(
                        electrode.get("probe_electrode")
                    ),
                    "brain_region": str(electrode["region_name"]),
                    "bad_channel": str(electrode.get("bad_channel")),
                    "reference_mode": reference_mode,
                    "reference_electrode_id": reference_electrode_id,
                    "is_reference": reference_electrode_id
                    == int(electrode["electrode_id"]),
                    "x": electrode_x,
                    "y": electrode_y,
                    "z": electrode.get("z"),
                    "rel_x": rel_x,
                    "rel_y": rel_y,
                    "rel_z": rel_z,
                    "contact_size": contact_size,
                    "plot_x": plot_x,
                    "plot_y": plot_y,
                    "coordinate_source": coord_source,
                }
            )
    return rows


def _layout_probe_contacts(plottable):
    """Set display coordinates while preserving each probe's physical geometry."""
    probe_ids = sorted(
        {row["probe_id"] for row in plottable},
        key=lambda probe_id: (probe_id is None, str(probe_id)),
    )
    by_probe = {
        probe_id: [row for row in plottable if row["probe_id"] == probe_id]
        for probe_id in probe_ids
    }
    multi_probe = len(probe_ids) > 1
    gap = None
    if multi_probe:
        import warnings

        # Scale by overall extent so single-column probes still get a gap.
        all_x = [row["plot_x"] for row in plottable]
        all_y = [row["plot_y"] for row in plottable]
        scale = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            1.0,
        )
        gap = 0.45 * scale
        cursor = 0.0
        for probe_id in probe_ids:
            probe_rows = by_probe[probe_id]
            xs = [row["plot_x"] for row in probe_rows]
            min_x, max_x = min(xs), max(xs)
            offset = cursor - min_x
            for row in probe_rows:
                row["display_x"] = row["plot_x"] + offset
            cursor += (max_x - min_x) + gap
        warnings.warn(
            f"plot_sort_group_geometry: {len(probe_ids)} probes present. "
            "Probe.Electrode rel_x/rel_y are per-probe coordinates, so the "
            "probes are laid out side-by-side along x (x positions are "
            "display-shifted per probe; within-probe geometry and y depths are "
            "unchanged).",
            UserWarning,
            stacklevel=3,
        )
    else:
        for row in plottable:
            row["display_x"] = row["plot_x"]

    return probe_ids, by_probe, gap


def _shade_probe_blocks(ax, multi_probe, gap, probe_ids, by_probe):
    """Shade alternating probe blocks in the display layout."""
    if multi_probe:
        band_pad = gap * 0.2
        for probe_index, probe_id in enumerate(probe_ids):
            if probe_index % 2:
                continue
            probe_xs = [row["display_x"] for row in by_probe[probe_id]]
            ax.axvspan(
                min(probe_xs) - band_pad,
                max(probe_xs) + band_pad,
                color="0.9",
                alpha=0.5,
                zorder=0,
            )


def _annotate_geometry(
    ax, plottable, by_probe, label_electrodes, max_per_column
):
    """Label sparse contacts and separate probe captions from their electrodes."""
    if label_electrodes is None:
        do_label = max_per_column <= _AUTO_LABEL_MAX_PER_COLUMN
    else:
        do_label = bool(label_electrodes)
    if do_label:
        for row in plottable:
            ax.annotate(
                str(row["electrode_id"]),
                (row["display_x"], row["plot_y"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )

    if len(by_probe) > 1:
        for probe_id, probe_rows in by_probe.items():
            center_x = sum(row["display_x"] for row in probe_rows) / len(
                probe_rows
            )
            top_y = max(row["plot_y"] for row in probe_rows)
            ax.annotate(
                str(probe_id),
                (center_x, top_y),
                # Clear of the electrode-id labels at offset (3, 3).
                xytext=(0, 22),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )


def plot_sort_group_geometry(
    nwb_file_name: str,
    *,
    ax=None,
    sort_group_ids: list[int] | tuple[int, ...] | set[int] | None = None,
    label_electrodes: bool | None = None,
    show_bad_channels: bool = True,
    show_reference: bool = True,
    title: str | None = None,
):
    """Plot a DB-backed geometry view of existing v2 sort groups.

    Contacts are colored by ``sort_group_id``, with bad channels and specific
    references overlaid. Reads Spyglass metadata only.

    Contacts sit where the recording stage places them: a ``tetrode_12.5``
    group on the 12.5 um square, any other group in the plane that separates
    every contact. A group the sort would refuse keeps its registered x-y.
    Multiple probes are laid out side by side along x.

    Parameters
    ----------
    nwb_file_name : str
        Session whose existing ``SortGroupV2`` rows should be visualized.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to draw into. A new figure/axes is created when
        omitted (the default).
    sort_group_ids : list of int or tuple of int or set of int, optional
        Subset of sort-group ids to display. All sort groups are shown
        when omitted (the default).
    label_electrodes : bool or None, optional
        Whether to annotate each contact with its ``electrode_id``. ``None``
        (default) labels only groups with at most
        ``_AUTO_LABEL_MAX_PER_COLUMN`` contacts.
    show_bad_channels : bool, optional
        If ``True``, overlay bad-channel members with red ``x`` markers.
        Defaults to ``True``.
    show_reference : bool, optional
        If ``True``, overlay ``reference_mode='specific'`` electrodes with a
        black star marker. Defaults to ``True``.
    title : str, optional
        Axes title. A default title naming the session is used when
        omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Warns
    -----
    UserWarning
        When the session spans more than one probe. ``Probe.Electrode``
        rel_x/rel_y are per-probe coordinates, so the probes are laid out
        side-by-side along x (x positions are display-shifted per probe;
        within-probe geometry and y depths are unchanged).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    rows = _sort_group_geometry_rows(nwb_file_name)
    if sort_group_ids is not None:
        wanted = {int(sort_group_id) for sort_group_id in sort_group_ids}
        rows = [row for row in rows if row["sort_group_id"] in wanted]

    if not rows:
        ax.text(
            0.5,
            0.5,
            "No SortGroupV2 rows",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    plottable = [
        row
        for row in rows
        if not _missing(row["plot_x"]) and not _missing(row["plot_y"])
    ]
    if not plottable:
        ax.text(
            0.5,
            0.5,
            "No plottable electrode geometry",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    # Each probe has its own coordinate frame, so offset probes along x.
    probe_ids, by_probe, gap = _layout_probe_contacts(plottable)
    multi_probe = len(probe_ids) > 1

    # Shrink markers for dense groups so stacked contacts stay separable.
    from collections import Counter

    contacts_per_column = Counter(row["sort_group_id"] for row in plottable)
    max_per_column = max(contacts_per_column.values())
    marker_s = max(8.0, min(50.0, 500.0 / max_per_column))
    bad_s = max(45.0, marker_s * 1.8)
    reference_s = max(90.0, marker_s * 3.0)

    _shade_probe_blocks(ax, multi_probe, gap, probe_ids, by_probe)

    cmap = plt.get_cmap("tab10")
    sort_group_ids = sorted({row["sort_group_id"] for row in plottable})
    for color_index, sort_group_id in enumerate(sort_group_ids):
        group_rows = [
            row
            for row in plottable
            if row["sort_group_id"] == sort_group_id and not row["is_reference"]
        ]
        color = cmap(color_index % cmap.N)
        ax.scatter(
            [row["display_x"] for row in group_rows],
            [row["plot_y"] for row in group_rows],
            s=marker_s,
            color=color,
            edgecolors="black",
            linewidths=0.35,
            alpha=0.85,
            label=f"sort_group_id {sort_group_id}",
        )

    if show_bad_channels:
        bad_rows = [
            row for row in plottable if str(row["bad_channel"]) == "True"
        ]
        if bad_rows:
            ax.scatter(
                [row["display_x"] for row in bad_rows],
                [row["plot_y"] for row in bad_rows],
                s=bad_s,
                marker="x",
                color="red",
                linewidths=1.2,
                label="bad channel",
            )

    if show_reference:
        reference_rows = [row for row in plottable if row["is_reference"]]
        if reference_rows:
            ax.scatter(
                [row["display_x"] for row in reference_rows],
                [row["plot_y"] for row in reference_rows],
                s=reference_s,
                marker="*",
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                label="specific reference",
            )

    _annotate_geometry(
        ax, plottable, by_probe, label_electrodes, max_per_column
    )

    coordinate_sources = {
        row["coordinate_source"]
        for row in plottable
        if row["coordinate_source"]
    }
    x_suffix = ", offset per probe" if multi_probe else ""
    probe_planes = {
        "xy" if source == "probe" else source.removeprefix("probe_")
        for source in coordinate_sources
        if source.startswith("probe")
    }
    if (
        coordinate_sources
        and len(probe_planes) == 1
        and all(source.startswith("probe") for source in coordinate_sources)
    ):
        (plane,) = probe_planes
        ax.set_xlabel(f"Probe rel_{plane[0]} (um{x_suffix})")
        ax.set_ylabel(f"Probe rel_{plane[1]} (um)")
    elif coordinate_sources == {"tetrode_12.5"}:
        ax.set_xlabel(f"tetrode_12.5 contact x (um{x_suffix})")
        ax.set_ylabel("tetrode_12.5 contact y (um)")
    elif coordinate_sources == {"electrode"}:
        ax.set_xlabel(f"Electrode x (um{x_suffix})")
        ax.set_ylabel("Electrode y (um)")
    else:
        ax.set_xlabel(f"x coordinate (um{x_suffix})")
        ax.set_ylabel("y coordinate (um)")

    missing_count = len(rows) - len(plottable)
    plot_title = title or f"Sort groups for {nwb_file_name}"
    if missing_count:
        plot_title = f"{plot_title} ({missing_count} contact(s) hidden)"
    ax.set_title(plot_title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)

    # Many groups: an in-axes legend would cover the contacts.
    if len(sort_group_ids) > 6:
        ax.legend(
            fontsize="small",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
    else:
        ax.legend(fontsize="small", loc="best")
    return ax
