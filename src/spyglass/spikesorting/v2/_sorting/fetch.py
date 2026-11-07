"""DB inputs for a ``Sorting`` populate (``Sorting.make_fetch``).

DataJoint tables are imported inside the functions.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, NamedTuple

from spyglass.spikesorting.v2._core.recipe_catalog import (
    waveform_params_for_preprocessing,
)
from spyglass.spikesorting.v2._sorting.analyzer import fetch_waveform_params
from spyglass.spikesorting.v2._artifacts.readers import (
    read_recording_artifact_valid_times,
)
from spyglass.spikesorting.v2._core.job_config import _resolved_job_kwargs
from spyglass.spikesorting.v2._params.sorter import validate_execution_params
from spyglass.spikesorting.v2._recording.unit_metadata import (
    sort_group_electrode_regions,
)
from spyglass.spikesorting.v2.exceptions import SchemaBypassError

if TYPE_CHECKING:
    from spyglass.spikesorting.v2._recording.source import SourceResolution
    from spyglass.spikesorting.v2.sorting import SortingFetched


def fetch_sorter_analyzer_inputs(key) -> tuple[dict, dict]:
    """Resolve the sorter row and execution kwargs before an analyzer build.

    Shared by populate, rebuild and recompute, so the computation never
    reads tables or ambient configuration.
    """
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    sorter_row = (
        SorterParameters
        & ((SortingSelection & key).proj("sorter", "sorter_params_name"))
    ).fetch1()
    return sorter_row, _resolved_job_kwargs(sorter_row["job_kwargs"])


def fetch_sorting_inputs(key) -> SortingFetched:
    """Read every DB input ``Sorting.make_compute`` needs.

    Parameters
    ----------
    key : dict
        Primary key restricting to one ``SortingSelection`` row.

    Returns
    -------
    SortingFetched
        DB inputs (source, anchor ``recording_id``, ``sel_row``,
        ``sorter_row``, anchor ``nwb_file_name``, ``obs_intervals``,
        display recipe, execution params) for the compute step.
    """
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingFetched,
        SortingSelection,
    )

    source = SortingSelection.resolve_source(key)

    sel_row = (SortingSelection & key).fetch1()
    # Lives on a part table, but downstream readers expect it on sel_row;
    # without it every artifact-backed sort would silently skip masking.
    sel_row["artifact_detection_id"] = (
        SortingSelection.resolve_artifact_detection(key)
    )
    sorter_row = (
        SorterParameters
        & {
            "sorter": sel_row["sorter"],
            "sorter_params_name": sel_row["sorter_params_name"],
        }
    ).fetch1()

    if (
        source.kind == "concatenated_recording"
        and sel_row.get("artifact_detection_id") is not None
    ):
        # Only a direct insert can reach this; concat members carry their masks.
        raise SchemaBypassError(
            "Sorting.make_fetch: concat source for sorting_id="
            f"{key.get('sorting_id')!r} carries artifact_detection_id="
            f"{sel_row['artifact_detection_id']!r}, but a concat sort "
            "owns its member masks through ConcatenatedRecordingSelection. "
            "The ArtifactDetectionSource "
            "part was inserted without SortingSelection.insert_selection "
            "(schema bypass) and cannot describe all concat members. Re-create the selection "
            "via insert_selection, or drop the stray "
            "ArtifactDetectionSource row."
        )

    anchor = resolve_sort_anchor(source)
    recording_id = anchor.recording_id
    nwb_file_name = anchor.nwb_file_name
    preprocessing_params_name = anchor.preprocessing_params_name
    if source.kind == "recording":
        # Firing rates need the artifact-removed observation intervals.
        # Without a detection, make_compute derives them from the recording.
        if sel_row.get("artifact_detection_id") is not None:
            obs_intervals = read_recording_artifact_valid_times(
                sel_row["artifact_detection_id"],
                nwb_file_name,
                caller="Sorting.make_fetch",
            )
        else:
            obs_intervals = None
        concat_statistics_spans = None
    else:  # concatenated_recording
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecording,
        )

        obs_intervals, concat_statistics_spans = (
            ConcatenatedRecording & source.key
        ).fetch1("obs_intervals", "statistics_spans")

    # Display recipe follows the preprocessing region; the name is persisted
    # so rebuilds read it back.
    display_waveform_params_name, _ = waveform_params_for_preprocessing(
        preprocessing_params_name
    )
    display_waveform_params = fetch_waveform_params(
        display_waveform_params_name
    )

    execution_params = validate_execution_params(
        sorter_row.get("execution_params")
    )

    sort_group_id, electrode_by_id, region_by_electrode = (
        fetch_unit_electrode_metadata(anchor.sort_group_id, nwb_file_name)
    )

    # After the concat schema-bypass check, which must fire first.
    traces = SortingSelection.resolve_effective_source(key).traces
    motion_correction_provenance = source_n_samples = None
    if traces.kind == "motion_corrected_recording":
        motion_correction_provenance, source_n_samples = (
            fetch_motion_correction(traces.key, sorter_row, source)
        )
    # Last, so a fetch that raises never rebuilds a file.
    traces_abs_path = SortingSelection.ensure_effective_traces(traces)

    return SortingFetched(
        source=source,
        recording_id=recording_id,
        sel_row=sel_row,
        sorter_row=sorter_row,
        nwb_file_name=nwb_file_name,
        obs_intervals=obs_intervals,
        display_waveform_params_name=display_waveform_params_name,
        display_waveform_params=display_waveform_params,
        execution_params=execution_params,
        sort_group_id=sort_group_id,
        electrode_by_id=electrode_by_id,
        region_by_electrode=region_by_electrode,
        concat_statistics_spans=concat_statistics_spans,
        traces=traces,
        traces_abs_path=traces_abs_path,
        motion_correction_provenance=motion_correction_provenance,
        source_n_samples=source_n_samples,
    )


def fetch_motion_correction(corrected_key, sorter_row, source):
    """DB inputs of a sort that reads a motion-corrected recording.

    Re-checks that the sorter does not correct motion itself, since its
    params or SpikeInterface defaults may have changed since selection.

    Parameters
    ----------
    corrected_key : dict
        ``{"motion_corrected_recording_id": ...}``.
    sorter_row : dict
        The sort's ``SorterParameters`` row.
    source : SourceResolution
        The sort's source.

    Returns
    -------
    tuple[dict, int]
        The provenance ids and recipe names, and the source's frame
        count (the concatenation's, or the one the estimate read from
        the recording).
    """
    from spyglass.spikesorting.v2._params.sorter import (
        reject_internal_motion_correction,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecordingSelection,
        MotionEstimate,
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    reject_internal_motion_correction(
        sorter_row["sorter"],
        sorter_row["params"],
        sorter_params_name=sorter_row["sorter_params_name"],
    )
    selection = (MotionCorrectedRecordingSelection & corrected_key).fetch1()
    estimate_key = {"motion_estimate_id": selection["motion_estimate_id"]}
    if source.kind == "concatenated_recording":
        source_n_samples = (ConcatenatedRecording & source.key).fetch1(
            "n_samples"
        )
    else:
        source_n_samples = (MotionEstimate & estimate_key).fetch1("n_samples")
    provenance = {
        "motion_corrected_recording_id": str(
            corrected_key["motion_corrected_recording_id"]
        ),
        "motion_estimate_id": str(selection["motion_estimate_id"]),
        "motion_estimation_params_name": (
            MotionEstimateSelection & estimate_key
        ).fetch1("motion_estimation_params_name"),
        "motion_interpolation_params_name": selection[
            "motion_interpolation_params_name"
        ],
    }
    return provenance, int(source_n_samples)


def fetch_unit_electrode_metadata(sort_group_id, nwb_file_name):
    """DB reads for the per-unit Electrode FK and brain region.

    ``electrode_by_id`` comes from the unjoined ``SortGroupElectrode`` so it
    is complete; ``region_by_electrode`` omits electrodes without a region.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    sort_group_id = int(sort_group_id)
    restriction = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
    }
    electrode_by_id = {
        int(row["electrode_id"]): row
        for row in (SortGroupV2.SortGroupElectrode & restriction).fetch(
            as_dict=True
        )
    }
    region_by_electrode = {
        int(row["electrode_id"]): str(row["region_name"])
        for row in sort_group_electrode_regions(restriction).fetch(
            "electrode_id", "region_name", as_dict=True
        )
    }
    return sort_group_id, electrode_by_id, region_by_electrode


class SortAnchor(NamedTuple):
    """The recording a sort anchors its session-level metadata to.

    A single-recording sort anchors to its own recording; a concat sort to
    its first frozen member. The anchor is the analysis-NWB parent and the
    per-unit ``Electrode`` FK source.
    """

    recording_id: uuid.UUID
    nwb_file_name: str
    sort_group_id: int
    preprocessing_params_name: str


def resolve_sort_anchor(source: SourceResolution) -> SortAnchor:
    """Resolve a sort source to its anchor recording.

    A concat sort reads the frozen ``MemberSnapshot``, never the live
    ``SessionGroup.Member`` set, so a group edit cannot re-point it.

    Parameters
    ----------
    source : SourceResolution
        The sort's source, from ``SortingSelection.resolve_source``.

    Returns
    -------
    SortAnchor
        The anchor ``recording_id``, ``nwb_file_name`` and ``sort_group_id``,
        and the source's ``preprocessing_params_name``.

    Raises
    ------
    SchemaBypassError
        If a concat selection has no ``MemberSnapshot`` rows.
    """
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    if source.kind == "recording":
        recording_id = source.key["recording_id"]
        nwb_file_name, sort_group_id, preprocessing_params_name = (
            RecordingSelection & {"recording_id": recording_id}
        ).fetch1("nwb_file_name", "sort_group_id", "preprocessing_params_name")
        return SortAnchor(
            recording_id=recording_id,
            nwb_file_name=str(nwb_file_name),
            sort_group_id=int(sort_group_id),
            preprocessing_params_name=str(preprocessing_params_name),
        )
    preprocessing_params_name = (
        ConcatenatedRecordingSelection & source.key
    ).fetch1("preprocessing_params_name")
    snapshot = (
        ConcatenatedRecordingSelection.MemberSnapshot & source.key
    ).fetch(as_dict=True, order_by="member_index", limit=1)
    if not snapshot:
        raise SchemaBypassError(
            "Sorting: concat selection "
            f"{dict(source.key)} has no MemberSnapshot rows; the frozen "
            "member set is written by insert_selection. Drop the selection "
            "and re-insert via insert_selection."
        )
    first_member = snapshot[0]
    return SortAnchor(
        recording_id=first_member["recording_id"],
        nwb_file_name=str(first_member["nwb_file_name"]),
        sort_group_id=int(first_member["sort_group_id"]),
        preprocessing_params_name=str(preprocessing_params_name),
    )


def resolve_anchor_nwb_file_name(key) -> str:
    """Return the analysis-NWB parent ``nwb_file_name`` for a sort.

    The ``nwb_file_name`` of :func:`resolve_sort_anchor`.

    Parameters
    ----------
    key : dict
        Restriction carrying ``sorting_id``.

    Returns
    -------
    str
        The anchor session's ``nwb_file_name``.
    """
    from spyglass.spikesorting.v2.sorting import SortingSelection

    return resolve_sort_anchor(
        SortingSelection.resolve_source(key)
    ).nwb_file_name
