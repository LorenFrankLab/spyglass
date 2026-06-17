"""Convenience orchestration across the spike sorting tables.

``run_v2_pipeline`` chains the recording -> artifact -> sort ->
curation stages into one call so notebook users can populate an
end-to-end single-session sort without writing the per-stage
insert_selection / populate boilerplate. The orchestrator focuses on
the minimum-viable single-session path; richer surfaces (metrics +
auto-curation, concat sorts, cross-session matching, UI hooks) come
in later versions.

Pipeline presets are Pydantic-validated bundles of Lookup-row names; the
orchestrator looks them up at first call. Three presets ship today:
    franklab_tetrode_mountainsort4
    franklab_tetrode_mountainsort5
    franklab_tetrode_clusterless_thresholder

The orchestrator is idempotent: re-running with the same inputs finds
existing rows via the insert_selection helpers and returns the same
manifest (with the same merge_id) without inserting duplicates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import pandas as pd


class _PipelinePreset(BaseModel):
    """A v2 pipeline preset: a bundle of Lookup-row names.

    The orchestrator consults this bundle once, then drives the
    ``insert_selection`` -> ``populate`` chain on each stage. ``params``
    Lookup row names must exist in the database before
    ``run_v2_pipeline`` is called; the preset itself does NOT insert
    Lookup rows (default rows are inserted explicitly by callers via
    ``*.insert_default()``).

    The human-facing fields (``intended_use``, ``threshold_units``,
    ``notes``) carry no runtime behavior; they describe the preset for
    ``describe_pipeline_presets()`` so a scientist can choose one without reading
    module source. They default to ``""`` so external presets need not
    supply them, but the built-ins below populate them.
    """

    model_config = ConfigDict(extra="forbid")
    preprocessing_params_name: str
    artifact_detection_params_name: str
    sorter: str
    sorter_params_name: str
    intended_use: str = ""  # one-line "when to reach for this pipeline preset"
    threshold_units: str = ""  # detection-threshold units (MAD mult. / µV)
    notes: str = ""  # key assumptions (probe geometry, sampling rate, etc.)


def list_pipeline_presets() -> list[str]:
    """Return the sorted pipeline-preset names accepted by ``run_v2_pipeline``.

    Notebook-discoverable accessor so users don't need to read the
    module source to learn what's available.

    Examples
    --------
    >>> from spyglass.spikesorting.v2.pipeline import list_pipeline_presets
    >>> list_pipeline_presets()
    ['franklab_tetrode_clusterless_thresholder', 'franklab_tetrode_mountainsort4', 'franklab_tetrode_mountainsort5']
    """
    return sorted(_PIPELINE_PRESETS)


def describe_pipeline_presets() -> "pd.DataFrame":
    """Return a table describing each pipeline preset accepted by ``run_v2_pipeline``.

    Companion to :func:`list_pipeline_presets` that adds the detail a scientist
    needs to choose a preset -- the sorter, the parameter-row names each
    stage uses, the intended use, and (a known footgun) the units of the
    detection threshold -- without reading module source. Pure and
    database-free: it reads only the in-module preset metadata and
    queries/inserts nothing. ``pandas`` is imported lazily so
    ``import spyglass.spikesorting.v2.pipeline`` stays cheap.

    Returns
    -------
    pandas.DataFrame
        One row per pipeline preset, sorted by name, with columns
        ``pipeline_preset``, ``sorter``, ``preprocessing_params_name``,
        ``artifact_detection_params_name``, ``sorter_params_name``,
        ``intended_use``, ``threshold_units``, and ``notes``. Call
        ``.to_dict("records")`` for raw rows.

    Examples
    --------
    >>> from spyglass.spikesorting.v2.pipeline import describe_pipeline_presets
    >>> describe_pipeline_presets()["pipeline_preset"].tolist()
    ['franklab_tetrode_clusterless_thresholder', 'franklab_tetrode_mountainsort4', 'franklab_tetrode_mountainsort5']
    """
    import pandas as pd

    columns = [
        "pipeline_preset",
        "sorter",
        "preprocessing_params_name",
        "artifact_detection_params_name",
        "sorter_params_name",
        "intended_use",
        "threshold_units",
        "notes",
    ]
    rows = [
        {
            "pipeline_preset": name,
            "sorter": pipeline_preset.sorter,
            "preprocessing_params_name": pipeline_preset.preprocessing_params_name,
            "artifact_detection_params_name": (
                pipeline_preset.artifact_detection_params_name
            ),
            "sorter_params_name": pipeline_preset.sorter_params_name,
            "intended_use": pipeline_preset.intended_use,
            "threshold_units": pipeline_preset.threshold_units,
            "notes": pipeline_preset.notes,
        }
        for name, pipeline_preset in sorted(_PIPELINE_PRESETS.items())
    ]
    return pd.DataFrame(rows, columns=columns)


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

    def _nullable_int(value):
        if value is None or pd.isna(value):
            return None
        return int(value)

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
    import pandas as pd

    from spyglass.common.common_device import Probe
    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_region import BrainRegion
    from spyglass.spikesorting.v2.recording import SortGroupV2

    def _missing(value) -> bool:
        return value is None or bool(pd.isna(value))

    def _nullable_int(value):
        if _missing(value):
            return None
        return int(value)

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
    import pandas as pd
    import matplotlib.pyplot as plt

    def _missing(value) -> bool:
        return value is None or bool(pd.isna(value))

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


_PIPELINE_PRESETS: dict[str, _PipelinePreset] = {
    "franklab_tetrode_mountainsort4": _PipelinePreset(
        preprocessing_params_name="default_franklab",
        artifact_detection_params_name="default",
        sorter="mountainsort4",
        sorter_params_name="franklab_tetrode_hippocampus_30kHz_ms4",
        intended_use=(
            "Frank-lab hippocampal tetrodes at 30 kHz; legacy MountainSort4 "
            "(parity with v1)."
        ),
        threshold_units="MAD multiplier",
        notes=(
            "MountainSort detect_threshold is a median-absolute-deviation "
            "multiplier on the per-channel noise floor, not an absolute "
            "voltage."
        ),
    ),
    "franklab_tetrode_mountainsort5": _PipelinePreset(
        preprocessing_params_name="default_franklab",
        artifact_detection_params_name="default",
        sorter="mountainsort5",
        sorter_params_name="franklab_tetrode_hippocampus_30kHz_ms5",
        intended_use=(
            "Frank-lab hippocampal tetrodes at 30 kHz; recommended default "
            "(current MountainSort5 settings)."
        ),
        threshold_units="MAD multiplier",
        notes=(
            "MountainSort detect_threshold is a median-absolute-deviation "
            "multiplier on the per-channel noise floor, not an absolute "
            "voltage."
        ),
    ),
    "franklab_tetrode_clusterless_thresholder": _PipelinePreset(
        preprocessing_params_name="default_franklab",
        artifact_detection_params_name="default",
        sorter="clusterless_thresholder",
        sorter_params_name="default",
        intended_use=(
            "Peak detection only (no clustering); feeds the clusterless "
            "decoding pipeline."
        ),
        threshold_units="µV (100 µV default)",
        notes=(
            "The 'default' clusterless SorterParameters row sets "
            "threshold_unit='uv' with detect_threshold=100, so the traces are "
            "scaled to microvolts before detection -- a true 100 µV threshold, "
            "not a MAD multiplier."
        ),
    ),
}


@dataclass(frozen=True)
class PreflightCheck:
    """One preflight check outcome: name, pass/fail, and the fix on fail."""

    name: str  # e.g. "session_exists", "sorter_installed"
    ok: bool
    fix: str  # empty when ok; the actionable fix when not ok


@dataclass(frozen=True)
class PreflightReport:
    """Result of ``preflight_v2_pipeline``: a pre-populate config check.

    Truthy when the configuration is runnable (``ok is True``), so a
    notebook can ``if not preflight_v2_pipeline(...): ...``. ``errors``
    lists each blocking problem with its fix; ``warnings`` holds
    non-blocking advisories; ``expected_ids`` carries the deterministic
    selection PKs the run would produce.

    Attributes
    ----------
    ok
        True when no blocking problem was found (``errors`` is empty).
    errors
        Blocking-problem messages; non-empty iff ``ok`` is False.
    warnings
        Non-blocking advisories (e.g. ``artifact_detection_params_name="none"``).
    resolved_pipeline_preset
        The pipeline-preset name that was checked.
    expected_ids
        The selection PKs a subsequent ``run_v2_pipeline`` would produce,
        each annotated with whether the row already exists, e.g.
        ``{"recording_id": {"id": UUID(...), "exists": False}, ...}``. IDs
        are computed DB-free via ``deterministic_id``; ``exists`` is a
        ``& pk`` restriction. Empty when the preset is unknown (the param
        names needed to derive the IDs are then unavailable). For an ``ok``
        report each ``id`` equals the PK ``run_v2_pipeline`` returns.
        ``curation_id`` is intentionally excluded: it is assigned by
        ``CurationV2.insert_curation``, not content-addressed.
    checks
        Per-check detail; every check runs (the report is complete, not
        first-failure-only).
    """

    ok: bool
    errors: list[str]
    warnings: list[str]
    resolved_pipeline_preset: str
    expected_ids: dict
    checks: list["PreflightCheck"]

    def __bool__(self) -> bool:
        """Return ``True`` when the configuration is runnable (``ok``)."""
        return self.ok


def preflight_v2_pipeline(
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    pipeline_preset: str = "franklab_tetrode_mountainsort5",
) -> PreflightReport:
    """Read-only pre-populate configuration check for ``run_v2_pipeline``.

    Verifies -- in ~1 s, inserting nothing and never calling ``populate``
    -- that every prerequisite a subsequent ``run_v2_pipeline(...,
    pipeline_preset=pipeline_preset)`` needs is in place: the session /
    interval / team / sort-group rows exist, the pipeline preset's
    parameter Lookup rows exist, and
    the sorter binary is installed. Returns a structured
    :class:`PreflightReport` instead of failing minutes into ``populate``
    with an opaque foreign-key or SpikeInterface error.

    Every check is a read-only restriction (``& {...}``) or a pure call;
    all checks run even after one fails, so the report lists every problem
    at once. An unknown ``pipeline_preset`` short-circuits before any database
    access (the later checks need the resolved param names).

    Parameters
    ----------
    nwb_file_name, sort_group_id, interval_list_name, team_name, pipeline_preset
        The same inputs as :func:`run_v2_pipeline`.

    Returns
    -------
    PreflightReport
        Truthy when the configuration is runnable. ``report.errors`` lists
        each blocking problem with the action to fix it;
        ``report.expected_ids`` holds the deterministic selection PKs the
        run would produce (see :class:`PreflightReport`).
    """
    checks: list[PreflightCheck] = []
    warnings: list[str] = []

    def _check(name: str, ok, fix: str) -> bool:
        ok = bool(ok)
        checks.append(PreflightCheck(name, ok, "" if ok else fix))
        return ok

    # 1. pipeline_preset_known. Short-circuit before any DB access on failure: the
    # remaining checks (and expected_ids) all derive from the resolved
    # bundle's param names, which are unknown for a bogus pipeline preset.
    if pipeline_preset not in _PIPELINE_PRESETS:
        _check(
            "pipeline_preset_known",
            False,
            f"unknown pipeline_preset {pipeline_preset!r}. Available pipeline presets: "
            f"{sorted(_PIPELINE_PRESETS)}. Call describe_pipeline_presets() to see what each "
            "one does.",
        )
        return PreflightReport(
            ok=False,
            errors=[c.fix for c in checks if not c.ok],
            warnings=warnings,
            resolved_pipeline_preset=pipeline_preset,
            expected_ids={},
            checks=checks,
        )
    _check("pipeline_preset_known", True, "")
    bundle = _PIPELINE_PRESETS[pipeline_preset]

    import spikeinterface.sorters as sis

    from spyglass.common import IntervalList, LabTeam, Session
    from spyglass.spikesorting.v2._selection_identity import (
        artifact_detection_identity_payload,
        deterministic_id,
        recording_identity_payload,
        sorting_identity_payload,
    )
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    sort_group_id = int(sort_group_id)

    # 2-5. Upstream session/interval/team/sort-group rows.
    _check(
        "session_exists",
        Session & {"nwb_file_name": nwb_file_name},
        f"session {nwb_file_name!r} is not ingested. Ingest it with "
        "insert_sessions(...) first.",
    )
    _check(
        "interval_exists",
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        },
        f"interval_list_name {interval_list_name!r} not found for "
        f"{nwb_file_name!r}. A full-session sort typically uses "
        "'raw data valid times'.",
    )
    _check(
        "team_exists",
        LabTeam & {"team_name": team_name},
        f"LabTeam {team_name!r} does not exist. Create it with "
        "LabTeam.insert1({'team_name': ..., 'team_description': ...}).",
    )
    _check(
        "sort_group_exists",
        SortGroupV2
        & {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id},
        f"SortGroupV2 sort_group_id={sort_group_id} not found for "
        f"{nwb_file_name!r}. Create sort groups first with "
        "SortGroupV2.set_group_by_shank(nwb_file_name=...).",
    )

    # 6-8. The preset's parameter Lookup rows.
    _check(
        "preprocessing_params_exist",
        PreprocessingParameters
        & {"preprocessing_params_name": bundle.preprocessing_params_name},
        f"PreprocessingParameters row {bundle.preprocessing_params_name!r} is "
        "missing. Run initialize_v2_defaults().",
    )
    _check(
        "artifact_detection_params_exist",
        ArtifactDetectionParameters
        & {
            "artifact_detection_params_name": bundle.artifact_detection_params_name
        },
        f"ArtifactDetectionParameters row {bundle.artifact_detection_params_name!r} "
        "is missing. Run initialize_v2_defaults().",
    )
    _check(
        "sorter_params_exist",
        SorterParameters
        & {
            "sorter": bundle.sorter,
            "sorter_params_name": bundle.sorter_params_name,
        },
        f"SorterParameters row (sorter={bundle.sorter!r}, "
        f"sorter_params_name={bundle.sorter_params_name!r}) is missing. "
        "Run initialize_v2_defaults().",
    )

    # 9. sorter_installed. Reuse the SAME strict gate insert_default uses:
    # the internal clusterless_thresholder (_NON_SI_SORTERS) is never an SI
    # binary and is always available; otherwise the sorter must be in
    # installed_sorters(), distinguishing "known but not installed" from
    # "misspelled / unknown" for the fix message.
    if (
        bundle.sorter in SorterParameters._NON_SI_SORTERS
        or bundle.sorter in set(sis.installed_sorters())
    ):
        _check("sorter_installed", True, "")
    elif bundle.sorter in set(sis.available_sorters()):
        _check(
            "sorter_installed",
            False,
            f"sorter {bundle.sorter!r} is a known SpikeInterface sorter but "
            "its binary/runtime is not installed here "
            "(spikeinterface.sorters.installed_sorters()). Install it, or "
            "pick a preset whose sorter is installed.",
        )
    else:
        _check(
            "sorter_installed",
            False,
            f"sorter {bundle.sorter!r} is not a known SpikeInterface sorter "
            "(spikeinterface.sorters.available_sorters()) -- check the "
            "spelling or the preset.",
        )

    # Non-blocking advisory: the "none" artifact params are a no-op
    # pass-through (no masking). "default" performs real amplitude-threshold
    # detection and is the legitimate built-in choice, so it is NOT warned.
    if bundle.artifact_detection_params_name == "none":
        warnings.append(
            "artifact_detection_params_name='none': no artifact masking will be "
            "applied for this run."
        )

    # expected_ids: the deterministic selection PKs this run would produce.
    # Pure hashes of (preset params + inputs) via the SAME payload builders
    # insert_selection uses, so they cannot drift; computable once the
    # preset resolves, regardless of whether the rows exist yet. ``exists``
    # is a read-only & pk check.
    recording_id = deterministic_id(
        "recording",
        recording_identity_payload(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
                "interval_list_name": interval_list_name,
                "preprocessing_params_name": bundle.preprocessing_params_name,
                "team_name": team_name,
            }
        ),
    )
    artifact_detection_id = deterministic_id(
        "artifact_detection",
        artifact_detection_identity_payload(
            artifact_detection_params_name=bundle.artifact_detection_params_name,
            recording_id=recording_id,
        ),
    )
    sorting_id = deterministic_id(
        "sorting",
        sorting_identity_payload(
            recording_id=recording_id,
            sorter=bundle.sorter,
            sorter_params_name=bundle.sorter_params_name,
            artifact_detection_id=artifact_detection_id,
        ),
    )
    expected_ids = {
        "recording_id": {
            "id": recording_id,
            "exists": bool(RecordingSelection & {"recording_id": recording_id}),
        },
        "artifact_detection_id": {
            "id": artifact_detection_id,
            "exists": bool(
                ArtifactDetectionSelection
                & {"artifact_detection_id": artifact_detection_id}
            ),
        },
        "sorting_id": {
            "id": sorting_id,
            "exists": bool(SortingSelection & {"sorting_id": sorting_id}),
        },
    }

    errors = [c.fix for c in checks if not c.ok]
    return PreflightReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        resolved_pipeline_preset=pipeline_preset,
        expected_ids=expected_ids,
        checks=checks,
    )


# Closed vocabulary for the per-stage ``*_status`` manifest keys. A stage is
# ``"computed"`` when its row did not exist before this call and populate /
# insert_curation created it this call; ``"reused"`` when the row already
# existed and the call no-opped. Test code asserts each status is a member.
_STAGE_STATUSES = frozenset({"computed", "reused"})


def _run_stage(
    stage: str, exists: bool, work, partial: dict
) -> tuple[Any, str, float]:
    """Time a pipeline stage's ``work()``; classify it; wrap failures.

    ``exists`` is the result of a pre-``work`` existence check on the stage's
    output row, so the status is ``"reused"`` when the row was already present
    (``work`` no-ops) and ``"computed"`` otherwise. ``work`` is a zero-arg
    closure over the stage's ``populate`` / ``insert_curation`` call; its
    return value is passed back (the curation stage needs the returned key).
    A failure is re-raised as a chained :class:`PipelineStageError` carrying a
    snapshot of ``partial`` (the manifest accumulated from earlier stages) so
    the caller sees which stage broke and what was already built.

    Returns ``(work_result, status, seconds)`` where ``seconds`` is monotonic
    wall-clock spent in ``work`` THIS call (≈0 on a reused no-op), not
    cumulative compute cost.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineStageError

    status = "reused" if exists else "computed"
    start = time.perf_counter()
    try:
        result = work()
    except Exception as exc:  # noqa: BLE001 - re-raised as typed + chained
        raise PipelineStageError(stage, dict(partial), str(exc)) from exc
    return result, status, time.perf_counter() - start


def run_v2_pipeline(
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    pipeline_preset: str = "franklab_tetrode_mountainsort5",
    description: str = "",
    require_units: bool = False,
    preflight: bool = True,
) -> dict[str, Any]:
    """End-to-end single-session sort: recording -> artifact detection -> sort -> curation.

    Chains the v2 ``insert_selection`` + ``populate`` calls into one
    call. Idempotent: re-running with the same inputs returns the same
    manifest (same merge_id, same intermediate PKs) without
    duplicating rows.

    Prerequisites (set these up first, in order)
    --------------------------------------------
    1. ``initialize_v2_defaults()`` -- seed the default Lookup rows.
    2. ``LabTeam`` row for ``team_name`` -- the owning team must already
       exist in ``common.LabTeam``.
    3. ``SortGroupV2.set_group_by_shank(nwb_file_name=...)`` (or
       ``set_group_by_electrode_table_column``) -- sort-group structure is
       session-specific user input the orchestrator does not auto-create.

    So a single-session sort is ~4 user touchpoints (the three setup steps
    above plus this call), not 2: this orchestrator collapses the per-stage
    ``insert_selection`` / ``populate`` boilerplate, not the upstream
    session/team/sort-group setup. With ``preflight=True`` (the default)
    this call verifies those prerequisites in ~1 s before any populate and
    raises ``PreflightError`` with the exact fix if one is missing; call
    ``preflight_v2_pipeline(...)`` directly to inspect the report without
    running.

    Parameters
    ----------
    nwb_file_name
        Session whose data will be sorted. The session must already be
        ingested via ``insert_sessions``.
    sort_group_id
        ID of an existing ``SortGroupV2`` row for this session.
        Callers create sort groups via
        ``SortGroupV2.set_group_by_shank`` (or
        ``set_group_by_electrode_table_column``) before calling this
        helper; the orchestrator does not auto-create them because
        sort-group structure is session-specific user input.
    interval_list_name
        Name of the IntervalList row to sort. Typically ``"raw data
        valid times"`` for a full-session sort.
    team_name
        LabTeam owning the sort. Must already exist in
        ``common.LabTeam``.
    pipeline_preset
        Pipeline-preset name from ``_PIPELINE_PRESETS``. Three presets ship today:
        ``franklab_tetrode_mountainsort4``,
        ``franklab_tetrode_mountainsort5`` (default), and
        ``franklab_tetrode_clusterless_thresholder``. Call
        ``describe_pipeline_presets()`` for a table of what each one does (sorter,
        parameter rows, intended use, and threshold units), or
        ``list_pipeline_presets()`` for just the names.
    description
        Free-text description passed to ``CurationV2.insert_curation``.
    require_units
        If False (default), a sort that finds zero units still produces
        an EMPTY (but real) curation + merge row, with a loud warning --
        zero units is a legitimate result on a quiet shank, and the empty
        row lets downstream code treat it like any other
        ``SpikeSortingOutput`` row. If True, a zero-unit sort raises
        ``ZeroUnitSortError`` instead (for callers that treat zero units
        as a hard error).
    preflight
        If True (default), run ``preflight_v2_pipeline`` first as a fast,
        read-only check that the session / interval / team / sort-group
        rows, the pipeline preset's parameter rows, and the sorter binary are all
        present; a failure raises ``PreflightError`` (with the exact fix)
        before any populate. Pass ``preflight=False`` to skip the check and
        attempt the run directly (e.g. to see the raw underlying error).

    Returns
    -------
    dict
        Manifest with the following stage keys:
            ``pipeline_preset``          : the pipeline-preset name
            ``recording_id``             : RecordingSelection PK
            ``artifact_detection_id``    : ArtifactDetectionSelection PK
            ``sorting_id``               : SortingSelection PK
            ``curation_id``              : CurationV2 PK
            ``merge_id``                 : SpikeSortingOutput master PK
            ``n_units``                  : unit count (0 on a zero-unit sort)
        Downstream consumers should key off ``merge_id``. A zero-unit
        sort yields an empty curation/merge row (matching v1's empty
        Units table), not ``None``, so the result is always
        merge-keyable.

        Plus per-stage observability keys (additive; the keys above are
        unchanged):
            ``recording_status`` / ``artifact_detection_status`` /
            ``sorting_status`` / ``curation_status`` : ``"computed"`` if the stage did work
                this call, ``"reused"`` if its row already existed and the
                call no-opped (see ``_STAGE_STATUSES``).
            ``stage_seconds``     : dict of monotonic wall-clock seconds
                spent per stage (keys ``"recording"`` / ``"artifact_detection"``
                / ``"sorting"`` / ``"curation"``) **this call** -- ≈0 on an
                idempotent re-run, NOT cumulative compute cost.
            ``warnings``          : list of human-readable advisories raised
                during the run (e.g. the zero-unit message); empty when
                clean.
        Two identical calls return equal manifests except for
        ``stage_seconds`` and the ``*_status`` values (the second reports
        ``"reused"``), inserting no duplicate rows.

    Raises
    ------
    PipelineInputError
        If ``pipeline_preset`` is not a known name.
    PreflightError
        If ``preflight=True`` and a prerequisite is missing (the message
        lists every failed check and its fix). Bypass with
        ``preflight=False``.
    PipelineStageError
        If a compute stage's ``populate`` / ``insert_curation`` fails. Names
        the failing stage and carries the partial manifest of the stages that
        completed before it (the original error is chained). Only the compute
        stages are wrapped; an error from the cheap ``insert_selection``
        prelude surfaces as its own native exception (e.g.
        ``DuplicateSelectionError``).
    ZeroUnitSortError
        If the sort finds zero units and ``require_units=True``.
    ValueError
        If a required parameter Lookup row is missing (e.g.
        ``PreprocessingParameters`` / ``SorterParameters`` defaults not
        installed); the insert helpers translate the would-be
        foreign-key error into this clear message. Run
        ``initialize_v2_defaults()`` first.
    datajoint.errors.IntegrityError
        If an upstream sort group / session / interval list / team does
        not exist when ``preflight=False`` -- the foreign-key violation
        surfaces untranslated. ``preflight=True`` catches these earlier
        as a ``PreflightError`` with the exact fix.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        PipelineInputError,
        PreflightError,
        ZeroUnitSortError,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )
    from spyglass.utils import logger

    if pipeline_preset not in _PIPELINE_PRESETS:
        raise PipelineInputError(
            f"run_v2_pipeline: unknown pipeline_preset {pipeline_preset!r}. "
            f"Available pipeline presets: {sorted(_PIPELINE_PRESETS)}. "
            "Call spyglass.spikesorting.v2.pipeline.describe_pipeline_presets() to see "
            "what each preset does, or list_pipeline_presets() for just the names."
        )
    bundle = _PIPELINE_PRESETS[pipeline_preset]

    # Fail fast: a read-only config check before any insert/populate, so a
    # missing team / interval / sort group / param row / sorter binary
    # surfaces in ~1 s with the exact fix, not minutes into populate() with
    # an opaque FK or SpikeInterface error. Bypass with preflight=False.
    if preflight:
        report = preflight_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name=interval_list_name,
            team_name=team_name,
            pipeline_preset=pipeline_preset,
        )
        if not report.ok:
            raise PreflightError("\n".join(report.errors))

    # Per-stage observability. For each stage: derive computed-vs-reused from
    # an existence check on the output row BEFORE populate, time the
    # populate/insert with a monotonic clock, and on failure raise a stage-
    # aware PipelineStageError carrying the manifest built so far. The stable
    # manifest keys are stable; *_status / stage_seconds / warnings are
    # additive. DataJoint's ``populate()`` is idempotent (no-ops on present
    # rows), so no separate guard is needed before each call.
    manifest: dict[str, Any] = {"pipeline_preset": pipeline_preset}
    stage_seconds: dict[str, float] = {}
    warnings_list: list[str] = []

    recording_key = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(sort_group_id),
            "interval_list_name": interval_list_name,
            "preprocessing_params_name": bundle.preprocessing_params_name,
            "team_name": team_name,
        }
    )
    _, manifest["recording_status"], stage_seconds["recording"] = _run_stage(
        "recording",
        bool(Recording & recording_key),
        lambda: Recording.populate(recording_key, reserve_jobs=False),
        manifest,
    )
    manifest["recording_id"] = recording_key["recording_id"]

    artifact_detection_key = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": recording_key["recording_id"],
            "artifact_detection_params_name": bundle.artifact_detection_params_name,
        }
    )
    (
        _,
        manifest["artifact_detection_status"],
        stage_seconds["artifact_detection"],
    ) = _run_stage(
        "artifact_detection",
        bool(ArtifactDetection & artifact_detection_key),
        lambda: ArtifactDetection.populate(
            artifact_detection_key, reserve_jobs=False
        ),
        manifest,
    )
    manifest["artifact_detection_id"] = artifact_detection_key[
        "artifact_detection_id"
    ]

    sorting_key = SortingSelection.insert_selection(
        {
            "recording_id": recording_key["recording_id"],
            "sorter": bundle.sorter,
            "sorter_params_name": bundle.sorter_params_name,
            "artifact_detection_id": artifact_detection_key[
                "artifact_detection_id"
            ],
        }
    )
    _, manifest["sorting_status"], stage_seconds["sorting"] = _run_stage(
        "sorting",
        bool(Sorting & sorting_key),
        lambda: Sorting.populate(sorting_key, reserve_jobs=False),
        manifest,
    )
    manifest["sorting_id"] = sorting_key["sorting_id"]

    # Zero units is a legitimate result on a quiet shank. Unless the
    # caller set require_units=True, proceed to build an empty (but real)
    # curation + merge row so the result is merge-keyable like any other.
    n_units = int((Sorting & sorting_key).fetch1("n_units"))
    if n_units == 0:
        recording_id = recording_key["recording_id"]
        if require_units:
            raise ZeroUnitSortError(
                "run_v2_pipeline: sort found zero units for "
                f"recording_id={recording_id} (sorting_id="
                f"{sorting_key['sorting_id']}); require_units=True. Check "
                "detect_threshold / the artifact mask, or call with "
                "require_units=False to accept the empty result."
            )
        # Fall through to the normal curation + merge insert. A
        # zero-unit sort yields an EMPTY (but real) curation + merge row
        # -- matching v1, which writes an empty Units table -- so
        # downstream consumers treat it like any other
        # SpikeSortingOutput row instead of special-casing a None
        # merge_id. The warning is both logged (console) and recorded on
        # the manifest's ``warnings`` list (programmatic access).
        zero_unit_warning = (
            "run_v2_pipeline: zero units for recording_id="
            f"{recording_id} (sorting_id={sorting_key['sorting_id']}); "
            "writing an EMPTY curation + merge row. Check "
            "detect_threshold / the artifact mask if you expected output."
        )
        logger.warning(zero_unit_warning)
        warnings_list.append(zero_unit_warning)

    # Record the now-known stable ``n_units`` and the ``warnings`` before the
    # curation stage runs, so a curation-stage failure's partial manifest
    # carries them (not just the pre-sorting keys).
    manifest["n_units"] = n_units
    manifest["warnings"] = warnings_list

    # Idempotent curation: ``insert_curation`` owns the root-reuse logic.
    # With ``reuse_existing=True`` it returns the canonical (lowest
    # curation_id) existing root if one is present -- deterministically,
    # and through the same source-part / guard / merge-registration path a
    # fresh insert uses -- otherwise it stages a fresh root. Routing through
    # it (rather than a raw fetch-or-insert here) avoids bypassing that
    # guard and silently reusing a root whose description/labels differ.
    # ``curation_id`` is not content-addressed, so classify reused/computed
    # from whether a root curation already exists for this sorting (the same
    # check insert_curation's root-reuse path uses).
    curation_exists = bool(
        CurationV2
        & {"sorting_id": sorting_key["sorting_id"], "parent_curation_id": -1}
    )

    def _curate_and_register():
        curation_key = CurationV2.insert_curation(
            sorting_key=sorting_key,
            labels={},
            parent_curation_id=-1,
            description=(
                description
                or f"run_v2_pipeline pipeline_preset={pipeline_preset}"
            ),
            reuse_existing=True,
        )
        # The CurationV2 part on the merge table is auto-registered inside
        # insert_curation (atomically), so the merge_id read-back is part of
        # the curation stage: a reused root whose registration is missing
        # (e.g. deleted out-of-band) then surfaces as a stage-aware
        # PipelineStageError carrying the partial manifest, not a raw fetch1.
        merge_id = (SpikeSortingOutput.CurationV2 & curation_key).fetch1(
            "merge_id"
        )
        return curation_key, merge_id

    (curation_key, merge_id), curation_status, curation_seconds = _run_stage(
        "curation", curation_exists, _curate_and_register, manifest
    )
    manifest["curation_status"] = curation_status
    stage_seconds["curation"] = curation_seconds
    manifest["curation_id"] = curation_key["curation_id"]
    manifest["merge_id"] = merge_id
    manifest["stage_seconds"] = stage_seconds
    return manifest
