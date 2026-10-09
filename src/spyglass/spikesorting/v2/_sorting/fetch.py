"""DB inputs for a ``Sorting`` populate.

:func:`fetch_sorting_inputs` is the body of ``Sorting.make_fetch``. It resolves
the sort's source and its anchor recording (the sort's own recording, or the
first frozen member of a concatenation), the sorter and display-analyzer
recipes, the execution backend, the per-unit electrode metadata, a
motion-corrected recording's provenance, and the effective traces file
(rebuilt here if missing). :func:`resolve_anchor_nwb_file_name` is the body of
the public ``Sorting.resolve_anchor_nwb_file_name``.

Imports without the DB layer: the DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spyglass.spikesorting.v2._core.recipe_catalog import (
    waveform_params_for_preprocessing,
)
from spyglass.spikesorting.v2._sorting.analyzer import fetch_waveform_params

if TYPE_CHECKING:
    from spyglass.spikesorting.v2.sorting import SortingFetched


def fetch_sorter_analyzer_inputs(key) -> tuple[dict, dict]:
    """Resolve the sorter row and execution kwargs before an analyzer build.

    Rebuild and recompute adapters use this same database boundary as populate
    callers: the computation function receives complete inputs and never
    consults tables or ambient configuration itself.
    """
    from spyglass.spikesorting.v2._core.job_config import _resolved_job_kwargs
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
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingFetched,
        SortingSelection,
    )

    source = SortingSelection.resolve_source(key)

    sel_row = (SortingSelection & key).fetch1()
    # The artifact-detection pass lives on the zero-or-one
    # ``ArtifactDetectionSource`` part, not a nullable
    # ``artifact_detection_id`` FK on the master, so ``sel_row`` does not
    # carry an ``artifact_detection_id`` key. Resolve it once here and stash it
    # on ``sel_row`` so the
    # downstream readers (obs_intervals derivation below,
    # make_compute's artifact-mask gate, rebuild_analyzer_folder)
    # see the artifact-detection id without re-querying. Without this the
    # ``sel_row.get("artifact_detection_id")`` reads would always be None
    # and every artifact-backed sort would silently skip artifact masking.
    # (Concat member masks are already materialized, so this is None there.)
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

    # Resolve the anchor recording_id / nwb file / preprocessing recipe.
    # Single-recording: the sort's own RecordingSelection. Concat: the
    # first member (deterministic parent anchor). Valid observation times
    # are resolved separately from this metadata anchor.
    if source.kind == "recording":
        recording_id = source.key["recording_id"]
        nwb_file_name, preprocessing_params_name = (
            RecordingSelection & {"recording_id": recording_id}
        ).fetch1("nwb_file_name", "preprocessing_params_name")
        # Pre-fetch the observation-interval window so ``_write_units_nwb``
        # can write ``obs_intervals=`` on every ``add_unit`` call.
        # Downstream firing-rate computations need the artifact-removed
        # valid_times to know which segments of the recording the sort
        # actually observed -- without it the units NWB looks like the unit
        # was observed across the full session even where the artifact mask
        # blanked the signal. When ``artifact_detection_id`` is unset,
        # make_compute derives the retained recording intervals, preserving
        # gaps between disjoint source intervals.
        if sel_row.get("artifact_detection_id") is not None:
            # Route through the strict ownership helper instead of
            # fetching the IntervalList directly by reconstructed name, and
            # raise clearly if this recording's intervals are absent rather
            # than feeding a dict to the mask.
            from spyglass.spikesorting.v2._artifacts.readers import (
                read_recording_artifact_valid_times,
            )

            obs_intervals = read_recording_artifact_valid_times(
                sel_row["artifact_detection_id"],
                nwb_file_name,
                caller="Sorting.make_fetch",
            )
        else:
            obs_intervals = None
        concat_statistics_spans = None
    else:  # concatenated_recording
        # Concat artifacts are masked before concatenation and carry
        # their own member detection provenance and kept intervals.
        # ``insert_selection`` rejects a concat source carrying an
        # artifact_detection_id, but a direct insert of a
        # ConcatenatedRecordingSource + ArtifactDetectionSource pair can
        # bypass that. Re-assert here (the compute boundary) so the sort is
        # never run UNMASKED while a stray ArtifactDetectionSource row claims
        # an artifact pass -- raise rather than silently dropping the mask.
        if sel_row.get("artifact_detection_id") is not None:
            from spyglass.spikesorting.v2.exceptions import (
                SchemaBypassError,
            )

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
        recording_id, nwb_file_name, preprocessing_params_name = (
            resolve_concat_anchor(source.key)
        )
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecording,
        )

        obs_intervals, concat_statistics_spans = (
            ConcatenatedRecording & source.key
        ).fetch1("obs_intervals", "statistics_spans")

    # Resolve the DISPLAY analyzer recipe from the source preprocessing
    # recipe (region) -- hippocampus -> the 0.5/0.5 row, cortex -> the
    # 1.0/2.0 row, any other recipe -> the wider cortex fallback. The
    # concat source resolves the SAME recipe from its single shared
    # preprocessing recipe. Resolve the upstream params blob HERE;
    # make_compute builds with it and make_insert persists the name so every
    # later rebuild reads it back deterministically.
    display_waveform_params_name, _ = waveform_params_for_preprocessing(
        preprocessing_params_name
    )
    display_waveform_params = fetch_waveform_params(
        display_waveform_params_name
    )

    # Resolve + validate the sorter execution backend (local vs container)
    # here. make_compute passes the resolved dict to the sorter dispatch.
    from spyglass.spikesorting.v2._params.sorter import (
        validate_execution_params,
    )

    execution_params = validate_execution_params(
        sorter_row.get("execution_params")
    )

    sort_group_id, electrode_by_id, region_by_electrode = (
        fetch_unit_electrode_metadata(recording_id, nwb_file_name)
    )

    # Resolved after the concat schema-bypass check above, which must fire
    # before any source-row fetch.
    traces = SortingSelection.resolve_effective_source(key).traces
    motion_correction_provenance = source_n_samples = None
    if traces.kind == "motion_corrected_recording":
        motion_correction_provenance, source_n_samples = (
            fetch_motion_correction(traces.key, sorter_row, source)
        )
    # Last, so a fetch that raises never rebuilds a file. A missing traces
    # file is rebuilt here, so the second fetch finds it and resolves the
    # same path.
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

    Re-checks that the sorter does not correct motion itself (the params
    row or SpikeInterface's defaults may differ from when the selection
    was inserted).

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


def fetch_unit_electrode_metadata(recording_id, nwb_file_name):
    """DB reads for the per-unit Electrode FK + brain region (fetch stage).

    Resolved once here so ``make_compute`` performs no DB writes while
    building the ``Sorting.Unit`` rows (and matching NWB unit columns).
    Upstream inputs are resolved in ``make_fetch``; the only DB reads
    left in ``make_compute`` stage the units NWB (see
    :mod:`._recording_nwb`). ``electrode_by_id`` comes from the unjoined
    ``SortGroupElectrode`` so it stays complete (the row-construction key set
    is unchanged); ``region_by_electrode`` is a best-effort
    ``electrode_id -> brain region`` map (an electrode without a region
    simply has no entry). Anchored to ``recording_id`` / ``nwb_file_name``
    (the sort's own recording, or the first concat member).
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_region import BrainRegion
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        SortGroupV2,
    )

    sort_group_id = int(
        (RecordingSelection & {"recording_id": recording_id}).fetch1(
            "sort_group_id"
        )
    )
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
        for row in (
            (SortGroupV2.SortGroupElectrode & restriction)
            * Electrode
            * BrainRegion
        ).fetch("electrode_id", "region_name", as_dict=True)
    }
    return sort_group_id, electrode_by_id, region_by_electrode


def first_concat_member(source_key):
    """Return the concat anchor member row and its preprocessing recipe.

    The anchor is the FIRST frozen member (by ``member_index``) from the
    ``ConcatenatedRecordingSelection.MemberSnapshot`` -- never the live
    ``SessionGroup.Member`` set, so a later group edit cannot re-point an
    existing concat sort's anchor. The frozen row carries the anchor's
    ``nwb_file_name`` (the analysis parent) and ``recording_id`` (the
    per-unit ``Electrode`` FK); the preprocessing recipe is the concat
    selection's single shared one.

    Parameters
    ----------
    source_key : dict
        ``{"concat_recording_id": ...}`` from ``resolve_source``.

    Returns
    -------
    tuple[dict, str]
        ``(first_member_snapshot_row, preprocessing_params_name)``.
    """
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    concat_sel = (ConcatenatedRecordingSelection & source_key).fetch1()
    snapshot = (
        ConcatenatedRecordingSelection.MemberSnapshot & source_key
    ).fetch(as_dict=True, order_by="member_index", limit=1)
    if not snapshot:
        raise SchemaBypassError(
            "Sorting: concat selection "
            f"{dict(source_key)} has no MemberSnapshot rows; the frozen "
            "member set is written by insert_selection. Drop the selection "
            "and re-insert via insert_selection."
        )
    return snapshot[0], concat_sel["preprocessing_params_name"]


def resolve_concat_anchor(source_key):
    """Resolve a concat source to its anchor recording / NWB / preprocessing.

    Reads the anchor ``recording_id`` straight from the frozen snapshot (no
    ``RecordingSelection`` join needed -- the snapshot froze the resolved
    ``recording_id`` when the concat id was minted). The full multi-session
    provenance stays queryable through
    ``ConcatenatedRecordingSelection.MemberSnapshot``.

    Parameters
    ----------
    source_key : dict
        ``{"concat_recording_id": ...}`` from ``resolve_source``.

    Returns
    -------
    tuple[uuid.UUID, str, str]
        ``(anchor_recording_id, anchor_nwb_file_name,
        preprocessing_params_name)``.
    """
    first_member, preprocessing_params_name = first_concat_member(source_key)
    return (
        first_member["recording_id"],
        first_member["nwb_file_name"],
        preprocessing_params_name,
    )


def resolve_anchor_nwb_file_name(key) -> str:
    """Return the analysis-NWB parent ``nwb_file_name`` for a sort.

    Source-agnostic: a single-recording sort anchors to its own
    ``RecordingSelection``; a concat sort anchors to the FIRST frozen
    ``MemberSnapshot`` member (the deterministic parent the per-unit Electrode
    FK and the curated/analyzer NWBs all use). Centralizes the
    unwrap-to-nwb dispatch that several reporting / curation accessors need,
    so the "which member is the anchor" decision lives in exactly one place.

    Parameters
    ----------
    key : dict
        Restriction carrying ``sorting_id``.

    Returns
    -------
    str
        The anchor session's ``nwb_file_name``.
    """
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source = SortingSelection.resolve_source(key)
    if source.kind == "recording":
        return (RecordingSelection & source.key).fetch1("nwb_file_name")
    # NWB only -- read it off the anchor member row; no recording_id join.
    return first_concat_member(source.key)[0]["nwb_file_name"]
