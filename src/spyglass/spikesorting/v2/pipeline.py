"""Convenience orchestration across the spike sorting tables.

``run_v2_pipeline`` chains the recording -> artifact -> sort ->
curation stages into one call so notebook users can populate an
end-to-end single-session sort without writing the per-stage
insert_selection / populate boilerplate. The orchestrator focuses on
the minimum-viable single-session path; richer surfaces (metrics +
auto-curation, concat sorts, cross-session matching, UI hooks) come
in later versions.

Presets are Pydantic-validated bundles of Lookup-row names; the
orchestrator looks them up at first call. Three presets ship today:
    franklab_tetrode_mountainsort4
    franklab_tetrode_mountainsort5
    franklab_tetrode_clusterless_thresholder

The orchestrator is idempotent: re-running with the same inputs finds
existing rows via the insert_selection helpers and returns the same
manifest (with the same merge_id) without inserting duplicates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import pandas as pd


class _Preset(BaseModel):
    """A v2 pipeline preset: a bundle of Lookup-row names.

    The orchestrator consults this bundle once, then drives the
    ``insert_selection`` -> ``populate`` chain on each stage. ``params``
    Lookup row names must exist in the database before
    ``run_v2_pipeline`` is called; the preset itself does NOT insert
    Lookup rows (default rows are inserted explicitly by callers via
    ``*.insert_default()``).

    The human-facing fields (``intended_use``, ``threshold_units``,
    ``notes``) carry no runtime behavior; they describe the preset for
    ``describe_presets()`` so a scientist can choose one without reading
    module source. They default to ``""`` so external presets need not
    supply them, but the built-ins below populate them.
    """

    model_config = ConfigDict(extra="forbid")
    preproc_params_name: str
    artifact_params_name: str
    sorter: str
    sorter_params_name: str
    intended_use: str = ""  # one-line "when to reach for this preset"
    threshold_units: str = ""  # detection-threshold units (MAD mult. / µV)
    notes: str = ""  # key assumptions (probe geometry, sampling rate, etc.)


def list_presets() -> list[str]:
    """Return the sorted preset names accepted by ``run_v2_pipeline``.

    Notebook-discoverable accessor so users don't need to read the
    module source to learn what's available.

    Examples
    --------
    >>> from spyglass.spikesorting.v2.pipeline import list_presets
    >>> list_presets()
    ['franklab_tetrode_clusterless_thresholder',
     'franklab_tetrode_mountainsort4',
     'franklab_tetrode_mountainsort5']
    """
    return sorted(_PRESETS)


def describe_presets() -> "pd.DataFrame":
    """Return a table describing each preset accepted by ``run_v2_pipeline``.

    Companion to :func:`list_presets` that adds the detail a scientist
    needs to choose a preset -- the sorter, the parameter-row names each
    stage uses, the intended use, and (a known footgun) the units of the
    detection threshold -- without reading module source. Pure and
    database-free: it reads only the in-module preset metadata and
    queries/inserts nothing. ``pandas`` is imported lazily so
    ``import spyglass.spikesorting.v2.pipeline`` stays cheap.

    Returns
    -------
    pandas.DataFrame
        One row per preset, sorted by preset name, with columns
        ``preset``, ``sorter``, ``preproc_params_name``,
        ``artifact_params_name``, ``sorter_params_name``,
        ``intended_use``, ``threshold_units``, and ``notes``. Call
        ``.to_dict("records")`` for raw rows.

    Examples
    --------
    >>> from spyglass.spikesorting.v2.pipeline import describe_presets
    >>> describe_presets()["preset"].tolist()
    ['franklab_tetrode_clusterless_thresholder', 'franklab_tetrode_mountainsort4', 'franklab_tetrode_mountainsort5']
    """
    import pandas as pd

    columns = [
        "preset",
        "sorter",
        "preproc_params_name",
        "artifact_params_name",
        "sorter_params_name",
        "intended_use",
        "threshold_units",
        "notes",
    ]
    rows = [
        {
            "preset": name,
            "sorter": preset.sorter,
            "preproc_params_name": preset.preproc_params_name,
            "artifact_params_name": preset.artifact_params_name,
            "sorter_params_name": preset.sorter_params_name,
            "intended_use": preset.intended_use,
            "threshold_units": preset.threshold_units,
            "notes": preset.notes,
        }
        for name, preset in sorted(_PRESETS.items())
    ]
    return pd.DataFrame(rows, columns=columns)


_PRESETS: dict[str, _Preset] = {
    "franklab_tetrode_mountainsort4": _Preset(
        preproc_params_name="default_franklab",
        artifact_params_name="default",
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
    "franklab_tetrode_mountainsort5": _Preset(
        preproc_params_name="default_franklab",
        artifact_params_name="default",
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
    "franklab_tetrode_clusterless_thresholder": _Preset(
        preproc_params_name="default_franklab",
        artifact_params_name="default",
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


def run_v2_pipeline(
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    preset: str = "franklab_tetrode_mountainsort5",
    description: str = "",
    require_units: bool = False,
) -> dict[str, Any]:
    """End-to-end single-session sort: recording -> artifact -> sort -> curation.

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
    session/team/sort-group setup.

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
    preset
        Preset name from ``_PRESETS``. Three presets ship today:
        ``franklab_tetrode_mountainsort4``,
        ``franklab_tetrode_mountainsort5`` (default), and
        ``franklab_tetrode_clusterless_thresholder``. Call
        ``describe_presets()`` for a table of what each one does (sorter,
        parameter rows, intended use, and threshold units), or
        ``list_presets()`` for just the names.
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

    Returns
    -------
    dict
        Manifest with the following stage keys:
            ``preset``                : the preset name
            ``recording_id``          : RecordingSelection PK
            ``artifact_id``           : ArtifactSelection PK
            ``sorting_id``            : SortingSelection PK
            ``curation_id``           : CurationV2 PK
            ``merge_id``              : SpikeSortingOutput master PK
            ``n_units``                : unit count (0 on a zero-unit sort)
        Downstream consumers should key off ``merge_id``. A zero-unit
        sort yields an empty curation/merge row (matching v1's empty
        Units table), not ``None``, so the result is always
        merge-keyable.

    Raises
    ------
    PipelineInputError
        If ``preset`` is not a known name.
    ZeroUnitSortError
        If the sort finds zero units and ``require_units=True``.
    ValueError
        If the upstream sort group / session / interval list / team
        do not exist (raised by the underlying insert helpers).
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        PipelineInputError,
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

    if preset not in _PRESETS:
        raise PipelineInputError(
            f"run_v2_pipeline: unknown preset {preset!r}. "
            f"Available presets: {sorted(_PRESETS)}. "
            "Call spyglass.spikesorting.v2.pipeline.describe_presets() to see "
            "what each preset does, or list_presets() for just the names."
        )
    bundle = _PRESETS[preset]

    # DataJoint's ``populate()`` is idempotent (no-ops on present
    # rows), so no separate ``if not (X & pk)`` guards are needed
    # before each call.
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(sort_group_id),
            "interval_list_name": interval_list_name,
            "preproc_params_name": bundle.preproc_params_name,
            "team_name": team_name,
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)

    art_pk = ArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_params_name": bundle.artifact_params_name,
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": bundle.sorter,
            "sorter_params_name": bundle.sorter_params_name,
            "artifact_id": art_pk["artifact_id"],
        }
    )
    Sorting.populate(sort_pk, reserve_jobs=False)

    # Zero units is a legitimate result on a quiet shank. Unless the
    # caller set require_units=True, proceed to build an empty (but real)
    # curation + merge row so the result is merge-keyable like any other.
    n_units = int((Sorting & sort_pk).fetch1("n_units"))
    if n_units == 0:
        recording_id = rec_pk["recording_id"]
        if require_units:
            raise ZeroUnitSortError(
                "run_v2_pipeline: sort found zero units for "
                f"recording_id={recording_id} (sorting_id="
                f"{sort_pk['sorting_id']}); require_units=True. Check "
                "detect_threshold / the artifact mask, or call with "
                "require_units=False to accept the empty result."
            )
        # Fall through to the normal curation + merge insert. A
        # zero-unit sort yields an EMPTY (but real) curation + merge row
        # -- matching v1, which writes an empty Units table -- so
        # downstream consumers treat it like any other
        # SpikeSortingOutput row instead of special-casing a None
        # merge_id.
        logger.warning(
            "run_v2_pipeline: zero units for recording_id="
            f"{recording_id} (sorting_id={sort_pk['sorting_id']}); "
            "writing an EMPTY curation + merge row. Check "
            "detect_threshold / the artifact mask if you expected output."
        )

    # Idempotent curation: ``insert_curation`` owns the root-reuse logic.
    # With ``reuse_existing=True`` it returns the canonical (lowest
    # curation_id) existing root if one is present -- deterministically,
    # and through the same source-part / guard / merge-registration path a
    # fresh insert uses -- otherwise it stages a fresh root. Routing through
    # it (rather than a raw fetch-or-insert here) avoids bypassing that
    # guard and silently reusing a root whose description/labels differ. The
    # CurationV2 part on the merge table is auto-registered inside
    # insert_curation, so a reused row reuses its merge_id.
    curation_pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={},
        parent_curation_id=-1,
        description=description or f"run_v2_pipeline preset={preset}",
        reuse_existing=True,
    )

    merge_id = (SpikeSortingOutput.CurationV2 & curation_pk).fetch1("merge_id")

    return {
        "preset": preset,
        "recording_id": rec_pk["recording_id"],
        "artifact_id": art_pk["artifact_id"],
        "sorting_id": sort_pk["sorting_id"],
        "curation_id": curation_pk["curation_id"],
        "merge_id": merge_id,
        "n_units": n_units,
    }
