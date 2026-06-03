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

from typing import Any

from pydantic import BaseModel, ConfigDict


class _Preset(BaseModel):
    """A v2 pipeline preset: a bundle of Lookup-row names.

    The orchestrator consults this bundle once, then drives the
    ``insert_selection`` -> ``populate`` chain on each stage. ``params``
    Lookup row names must exist in the database before
    ``run_v2_pipeline`` is called; the preset itself does NOT insert
    Lookup rows (default rows are inserted explicitly by callers via
    ``*.insert_default()``).
    """

    model_config = ConfigDict(extra="forbid")
    preproc_params_name: str
    artifact_params_name: str
    sorter: str
    sorter_params_name: str


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


_PRESETS: dict[str, _Preset] = {
    "franklab_tetrode_mountainsort4": _Preset(
        preproc_params_name="default_franklab",
        artifact_params_name="default",
        sorter="mountainsort4",
        sorter_params_name="franklab_tetrode_hippocampus_30kHz_ms4",
    ),
    "franklab_tetrode_mountainsort5": _Preset(
        preproc_params_name="default_franklab",
        artifact_params_name="default",
        sorter="mountainsort5",
        sorter_params_name="franklab_tetrode_hippocampus_30kHz_ms5",
    ),
    "franklab_tetrode_clusterless_thresholder": _Preset(
        preproc_params_name="default_franklab",
        artifact_params_name="default",
        sorter="clusterless_thresholder",
        sorter_params_name="default",
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
        ``franklab_tetrode_clusterless_thresholder``.
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
            "Call run_v2_pipeline? to see the docstring, or check "
            "spyglass.spikesorting.v2.pipeline.list_presets()."
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

    # Idempotent curation: if a root (parent_curation_id=-1) curation
    # already exists for this sorting, reuse it; otherwise insert a
    # fresh one. The CurationV2 part on the merge table is auto-
    # registered inside insert_curation, so re-using a row reuses its
    # merge_id.
    existing_root = (CurationV2 & sort_pk & {"parent_curation_id": -1}).fetch(
        "KEY", as_dict=True
    )
    if existing_root:
        curation_pk = existing_root[0]
    else:
        curation_pk = CurationV2.insert_curation(
            sorting_key=sort_pk,
            labels={},
            parent_curation_id=-1,
            description=description or f"run_v2_pipeline preset={preset}",
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
