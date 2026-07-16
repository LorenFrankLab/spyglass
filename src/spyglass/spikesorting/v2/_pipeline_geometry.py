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


# Auto-label electrode ids only when each sort group has at most this many
# contacts (tetrode = 4, stereotrode = 2): sparse groups stay legible, while a
# dense polymer / Neuropixels column (>= ~16 per group) would be a wall of text.
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

    # Reference electrodes for 'specific'-reference groups are NOT sort-group
    # members (membership excludes them), so fetch them directly; the group loop
    # below appends each as a synthetic row so the is_reference star overlay has
    # a row to render.
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

    probe_restrictions = []
    for electrode in [*member_rows, *reference_electrode_by_id.values()]:
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
        # Append the (non-member) specific reference electrode so its row gets
        # is_reference=True below and renders as the star overlay; it is excluded
        # from the per-group electrode scatter in plot_sort_group_geometry.
        if reference_electrode_id in reference_electrode_by_id:
            group_electrodes = [
                *group_electrodes,
                reference_electrode_by_id[reference_electrode_id],
            ]
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
    label_electrodes: bool | None = None,
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

    Contact marker size auto-shrinks as a sort group gets dense (a linear
    polymer shank stacks many contacts in a narrow band) so contacts stay
    separable; a sparse tetrode keeps large markers. Electrode-id labels (the
    channel numbers) are shown automatically for sparse sort groups (tetrodes /
    stereotrodes) and hidden for dense probes where they would overlap into a
    wall of text; ``label_electrodes`` overrides this. Specific reference
    electrodes are marked with a star -- see ``show_reference`` for where that
    reference comes from; its channel id is in ``describe_sort_groups``.

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
    label_electrodes : bool or None, optional
        Whether to annotate each plotted contact with its ``electrode_id`` (the
        channel number). ``None`` (the default) auto-decides on column density:
        labels are shown when every sort group has at most
        ``_AUTO_LABEL_MAX_PER_COLUMN`` contacts (tetrodes / stereotrodes) and
        hidden for denser probes where they would overlap. ``True`` / ``False``
        force labels on / off.
    show_bad_channels : bool, optional
        If ``True``, overlay bad-channel members with red ``x`` markers.
        Defaults to ``True``.
    show_reference : bool, optional
        If ``True``, overlay ``reference_mode='specific'`` electrodes with a
        black star marker. Defaults to ``True``. This reference is the one each
        sort group inherited from ``Electrode.original_reference_electrode``
        (ultimately the ``trodes_to_nwb`` ``ref_elect_id`` metadata), unless it
        was overridden via ``set_group_by_shank(references=...)`` or
        ``reference_mode=...`` when the sort group was created. Groups with
        ``reference_mode`` of ``'none'`` or ``'global_median'`` have no star.
        The reference's channel id is available in ``describe_sort_groups``.
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
            stacklevel=2,
        )
    else:
        for row in plottable:
            row["display_x"] = row["plot_x"]

    # Marker size shrinks as a column gets dense so stacked contacts don't
    # overlap: a linear polymer shank packs many contacts into a narrow y band,
    # while the base size is tuned for a ~4-contact tetrode. Bounded so sparse
    # groups stay large/readable and dense ones stay separable.
    from collections import Counter

    contacts_per_column = Counter(row["sort_group_id"] for row in plottable)
    max_per_column = max(contacts_per_column.values())
    marker_s = max(8.0, min(50.0, 500.0 / max_per_column))
    bad_s = max(45.0, marker_s * 1.8)
    reference_s = max(90.0, marker_s * 3.0)

    # Shade alternating probe blocks so the side-by-side layout reads as
    # per-probe groups rather than one undifferentiated row of columns.
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

    # Electrode-id labels: shown automatically only when each sort group is
    # sparse enough to stay legible (tetrodes / stereotrodes) -- a dense polymer
    # or Neuropixels column would be a wall of overlapping text.
    # ``label_electrodes`` True/False forces the choice; ``None`` (default)
    # auto-decides on column density. When shown, the specific reference gets its
    # electrode_id like any other contact (the star already marks it), so no
    # separate reference label is needed.
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
                # Lifted well above the top contacts so the bold probe_id sits
                # clear of any electrode-id labels drawn on them (offset (3, 3))
                # when label_electrodes is on, rather than overlapping.
                xytext=(0, 22),
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

    # With many sort groups the in-axes legend covers the contacts; move it
    # outside to the right. A handful of groups (the tetrode case) reads fine
    # inside at "best".
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
