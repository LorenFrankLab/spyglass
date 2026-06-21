"""Sort-group geometry reporting + plotting, extracted from ``pipeline.py``.

Behavior-preserving: ``describe_sort_groups``, ``_sort_group_geometry_rows``,
and ``plot_sort_group_geometry`` (plus their ``_SORT_GROUP_COLUMNS`` constant)
move here verbatim. ``pipeline.py`` re-exports the two public functions so
notebook import paths are unchanged. Self-contained (no dependency on the
other pipeline submodules).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


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
    """True when ``value`` is ``None`` or a pandas/NumPy NA/NaN."""
    import pandas as pd

    return value is None or bool(pd.isna(value))


def _nullable_int(value):
    """Coerce a possibly-missing numeric to ``int``, or ``None`` if missing."""
    return None if _missing(value) else int(value)


def describe_sort_groups(nwb_file_name: str) -> "pd.DataFrame":
    """Return a notebook-friendly summary of v2 sort groups for a session.

    Use this after creating ``SortGroupV2`` rows, before choosing a
    ``sort_group_id`` for ``run_v2_pipeline``. The table surfaces the
    scientific context a user normally needs at that decision point:
    electrode membership, electrode groups, probe shanks, brain regions,
    bad-channel membership, and reference mode. The helper is read-only:
    it restricts existing ``SortGroupV2`` / ``Electrode`` / ``BrainRegion``
    rows and never creates sort groups.

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

    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_region import BrainRegion
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

    rows = []
    for master in sorted(
        master_rows, key=lambda row: int(row["sort_group_id"])
    ):
        sort_group_id = int(master["sort_group_id"])
        restriction = {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
        }
        electrode_rows = (
            (SortGroupV2.SortGroupElectrode & restriction)
            * Electrode
            * BrainRegion
        ).fetch(as_dict=True)

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


def _sort_group_geometry_rows(nwb_file_name: str) -> list[dict[str, Any]]:
    """Return DB-backed electrode geometry rows for sort-group plotting."""
    from spyglass.common.common_device import Probe
    from spyglass.common.common_ephys import Electrode
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

    member_rows = (
        (SortGroupV2.SortGroupElectrode & {"nwb_file_name": nwb_file_name})
        * Electrode
        * BrainRegion
    ).fetch(as_dict=True)

    probe_restrictions = []
    for electrode in member_rows:
        probe_key = {
            "probe_id": electrode.get("probe_id"),
            "probe_shank": electrode.get("probe_shank"),
            "probe_electrode": electrode.get("probe_electrode"),
        }
        if all(not _missing(value) for value in probe_key.values()):
            probe_restrictions.append(
                {
                    key: _nullable_int(value) if key != "probe_id" else value
                    for key, value in probe_key.items()
                }
            )
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
            (
                row
                for row in member_rows
                if int(row["sort_group_id"]) == sort_group_id
            ),
            key=lambda row: int(row["electrode_id"]),
        )
        for electrode in group_electrodes:
            rel_x = rel_y = rel_z = contact_size = None
            probe_key = {
                "probe_id": electrode.get("probe_id"),
                "probe_shank": electrode.get("probe_shank"),
                "probe_electrode": electrode.get("probe_electrode"),
            }
            if all(not _missing(value) for value in probe_key.values()):
                geometry = probe_geometry.get(
                    (
                        probe_key["probe_id"],
                        _nullable_int(probe_key["probe_shank"]),
                        _nullable_int(probe_key["probe_electrode"]),
                    )
                )
                if geometry:
                    rel_x = geometry["rel_x"]
                    rel_y = geometry["rel_y"]
                    rel_z = geometry["rel_z"]
                    contact_size = geometry["contact_size"]

            # Pick plot coordinates from a SINGLE source so plot_x/plot_y and
            # coordinate_source can never disagree (e.g. plot probe rel_x
            # against electrode y, or label "electrode" while plotting a probe
            # coord). Probe rel_x/rel_y are populated together, but pairing
            # here keeps the contract explicit if only one were present.
            electrode_x = electrode.get("x")
            electrode_y = electrode.get("y")
            if not _missing(rel_x) and not _missing(rel_y):
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


def plot_sort_group_geometry(
    nwb_file_name: str,
    *,
    ax=None,
    sort_group_ids: list[int] | tuple[int, ...] | set[int] | None = None,
    label_electrodes: bool = False,
    show_bad_channels: bool = True,
    show_reference: bool = True,
    title: str | None = None,
):
    """Plot a DB-backed geometry view of existing v2 sort groups.

    Use this immediately after ``describe_sort_groups`` and before choosing a
    ``sort_group_id``. Contacts are colored by ``sort_group_id``; bad-channel
    members and specific reference electrodes are overlaid when present. The
    helper reads Spyglass metadata only -- it does not open the raw recording or
    create SpikeInterface objects.

    When the session spans **more than one probe**, ``Probe.Electrode``
    rel_x/rel_y are each probe's own coordinate frame (all near the origin), so
    the probes are laid out **side-by-side** along x -- each probe's contacts
    are display-shifted into their own column (annotated with the ``probe_id``)
    and a ``UserWarning`` is emitted. y depths and within-probe geometry are
    unchanged; the underlying ``rel_x`` is not mutated.

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
    label_electrodes : bool, optional
        If ``True``, annotate each plotted contact with its
        ``electrode_id``. Defaults to ``False``.
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
        _, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

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

    # ``Probe.Electrode`` rel_x/rel_y are each probe's OWN coordinate frame
    # (every probe starts near the origin), so plotting multiple probes on one
    # shared axis would overlap them. Lay probes out side-by-side along x by
    # offsetting each probe's contacts into its own column; y (depth) is left
    # untouched. ``display_x`` carries the (possibly shifted) plot coordinate so
    # the raw per-probe ``plot_x`` is preserved on each row.
    probe_ids = sorted(
        {row["probe_id"] for row in plottable},
        key=lambda probe_id: (probe_id is None, str(probe_id)),
    )
    by_probe = {
        probe_id: [row for row in plottable if row["probe_id"] == probe_id]
        for probe_id in probe_ids
    }
    multi_probe = len(probe_ids) > 1
    if multi_probe:
        import warnings

        # Gap scales with the overall geometry extent so it is non-zero even
        # for single-column (linear) probes whose contacts share one rel_x.
        all_x = [row["plot_x"] for row in plottable]
        all_y = [row["plot_y"] for row in plottable]
        scale = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            1.0,
        )
        gap = 0.15 * scale
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
            stacklevel=2,
        )
    else:
        for row in plottable:
            row["display_x"] = row["plot_x"]

    cmap = plt.get_cmap("tab10")
    sort_group_ids = sorted({row["sort_group_id"] for row in plottable})
    for color_index, sort_group_id in enumerate(sort_group_ids):
        group_rows = [
            row for row in plottable if row["sort_group_id"] == sort_group_id
        ]
        color = cmap(color_index % cmap.N)
        ax.scatter(
            [row["display_x"] for row in group_rows],
            [row["plot_y"] for row in group_rows],
            s=50,
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
                s=90,
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
                s=150,
                marker="*",
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                label="specific reference",
            )

    if label_electrodes:
        for row in plottable:
            ax.annotate(
                str(row["electrode_id"]),
                (row["display_x"], row["plot_y"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )

    # Label each probe's column so the side-by-side layout is interpretable.
    if multi_probe:
        for probe_id in probe_ids:
            probe_rows = by_probe[probe_id]
            center_x = sum(row["display_x"] for row in probe_rows) / len(
                probe_rows
            )
            top_y = max(row["plot_y"] for row in probe_rows)
            ax.annotate(
                str(probe_id),
                (center_x, top_y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    coordinate_sources = {
        row["coordinate_source"]
        for row in plottable
        if row["coordinate_source"]
    }
    x_suffix = ", offset per probe" if multi_probe else ""
    if coordinate_sources == {"probe"}:
        ax.set_xlabel(f"Probe rel_x (um{x_suffix})")
        ax.set_ylabel("Probe rel_y (um)")
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

    ax.legend(fontsize="small", loc="best")
    return ax
