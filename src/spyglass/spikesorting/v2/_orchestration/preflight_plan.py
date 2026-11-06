"""The ids a run would compute or reuse, and its resource notes."""

from __future__ import annotations

import uuid
from typing import Any, TYPE_CHECKING

from spyglass.spikesorting.v2._orchestration.preflight_motion import (
    MotionRecipe,
    expected_source_ids,
    supplied_motion_estimate_problem,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _preflight_expected_ids(
    check: "Callable[[str, Any, str], bool]",
    bundle,
    *,
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    manual_excluded_times,
    motion_mode,
    motion_recipe: "MotionRecipe | None",
    motion_estimate_id,
) -> dict:
    """The selection PKs a run would produce, each with its existence flags.

    Also records the ``motion_estimate_applicable`` check for a supplied
    ``motion_estimate_id``, which needs the expected recording and artifact
    detection ids.

    Returns
    -------
    dict
        The :attr:`PreflightReport.expected_ids` mapping.
    """
    from spyglass.spikesorting.v2._core.selection_plan import (
        build_sorting_selection_plan,
    )
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    recording_id, artifact_detection_id = expected_source_ids(
        bundle,
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name=interval_list_name,
        team_name=team_name,
        manual_excluded_times=manual_excluded_times,
    )
    if motion_estimate_id is not None and motion_recipe is not None:
        from spyglass.spikesorting.v2._recording.source import SourceLineage

        supplied = supplied_motion_estimate_problem(
            motion_estimate_id,
            motion_recipe,
            source_lineage=SourceLineage(
                kind="recording",
                key={"recording_id": recording_id},
                artifact_detection_id=artifact_detection_id,
            ),
        )
        check("motion_estimate_applicable", supplied is None, supplied)
    motion_ids = _expected_motion_ids(
        motion_mode,
        motion_recipe,
        recording_id=recording_id,
        artifact_detection_id=artifact_detection_id,
        motion_estimate_id=motion_estimate_id,
    )
    corrected = motion_ids.get("motion_corrected_recording_id", {})
    if corrected.get("pending"):
        sorting_entry = _pending_id_entry(corrected["pending"])
    else:
        sorting_id = build_sorting_selection_plan(
            {
                "recording_id": recording_id,
                "sorter": bundle.sorter,
                "sorter_params_name": bundle.sorter_params_name,
                "artifact_detection_id": artifact_detection_id,
                "motion_corrected_recording_id": corrected.get("id"),
            }
        ).sorting_id
        sorting_entry = {
            "id": sorting_id,
            "exists": bool(SortingSelection & {"sorting_id": sorting_id}),
            "computed_exists": bool(Sorting & {"sorting_id": sorting_id}),
        }
    # Per stage, ``exists`` is whether the SELECTION row exists (the run
    # would reuse this PK) and ``computed_exists`` whether the COMPUTED
    # output row exists (the populate already ran -- a reused, near-zero-cost
    # stage). Distinguishing them tells a caller what work the run would
    # actually do: a selection can exist with its output not yet populated.
    return {
        "recording_id": {
            "id": recording_id,
            "exists": bool(RecordingSelection & {"recording_id": recording_id}),
            "computed_exists": bool(Recording & {"recording_id": recording_id}),
        },
        "artifact_detection_id": (
            {"id": None, "exists": False, "computed_exists": False}
            if artifact_detection_id is None
            else {
                "id": artifact_detection_id,
                "exists": bool(
                    RecordingArtifactSelection
                    & {"artifact_detection_id": artifact_detection_id}
                ),
                "computed_exists": bool(
                    RecordingArtifactDetection
                    & {"artifact_detection_id": artifact_detection_id}
                ),
            }
        ),
        **motion_ids,
        "sorting_id": sorting_entry,
    }


def _group_resource_notes(
    display_waveform_params_name: str,
    *,
    nwb_file_name: str,
    sort_group_id: int,
    effective_config: "dict | None",
) -> list[str]:
    """:func:`_resource_notes` for one sort group, read from its rows."""
    from spyglass.common import Raw
    from spyglass.spikesorting.v2.recording import SortGroupV2

    return _resource_notes(
        display_waveform_params_name,
        n_channels=len(
            SortGroupV2.SortGroupElectrode
            & {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}
        ),
        sampling_rate_hz=(
            float(
                (Raw & {"nwb_file_name": nwb_file_name}).fetch1("sampling_rate")
            )
            if Raw & {"nwb_file_name": nwb_file_name}
            else None
        ),
        effective_config=effective_config,
    )


def _pending_id_entry(reason: str) -> dict:
    """An ``expected_ids`` entry whose id cannot be derived yet, and why."""
    return {
        "id": None,
        "exists": False,
        "computed_exists": False,
        "pending": reason,
    }


def _expected_motion_ids(
    motion_mode,
    motion_recipe: "MotionRecipe | None",
    *,
    recording_id,
    artifact_detection_id,
    motion_estimate_id=None,
) -> dict:
    """Preview the motion selection ids a single-recording run would mint.

    The estimate id folds in the source recording's ``content_hash``, so it
    (and the corrected id and an ``"apply"`` sort id built on it) is known
    only once the ``Recording`` is computed; until then the entries carry
    ``id=None`` and a ``pending`` reason instead of a guess. A supplied
    ``motion_estimate_id`` (the saved estimate an ``"apply"`` run reuses) is
    known up front. Uses the DB-free derivations the selection inserts use,
    so a preview cannot drift from the insert.

    Returns
    -------
    dict
        ``{}`` for ``"off"`` or an unresolved recipe; otherwise
        ``motion_estimate_id`` (and, for ``"apply"``,
        ``motion_corrected_recording_id``) entries shaped like the other
        ``expected_ids`` entries.
    """
    if motion_mode == "off" or motion_recipe is None:
        return {}
    from spyglass.spikesorting.v2._motion.estimation import (
        motion_corrected_selection_identity,
        motion_estimate_selection_identity,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionCorrectedRecordingSelection,
        MotionEstimate,
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.recording import Recording

    names = ["motion_estimate_id"] + (
        ["motion_corrected_recording_id"] if motion_mode == "apply" else []
    )
    if motion_estimate_id is not None:
        estimate_id = uuid.UUID(str(motion_estimate_id))
    else:
        content_hashes = (Recording & {"recording_id": recording_id}).fetch(
            "content_hash"
        )
        if len(content_hashes) == 0:
            pending = (
                "the recording is computed (the motion estimate id includes "
                "its content hash)"
            )
            return {name: _pending_id_entry(pending) for name in names}
        estimate_id = motion_estimate_selection_identity(
            source_kind="recording",
            source_id=recording_id,
            source_content_hash=content_hashes[0],
            artifact_detection_id=artifact_detection_id,
            motion_estimation_params_name=motion_recipe.recipe[
                "motion_estimation_params_name"
            ],
            estimation_params=motion_recipe.estimation_params,
        ).selection_id
    estimate_key = {"motion_estimate_id": estimate_id}
    expected = {
        "motion_estimate_id": {
            "id": estimate_id,
            "exists": bool(MotionEstimateSelection & estimate_key),
            "computed_exists": bool(MotionEstimate & estimate_key),
        }
    }
    if motion_mode == "apply":
        corrected_id = motion_corrected_selection_identity(
            motion_estimate_id=estimate_id,
            motion_interpolation_params_name=motion_recipe.recipe[
                "motion_interpolation_params_name"
            ],
            interpolation_params=motion_recipe.interpolation_params,
        ).selection_id
        corrected_key = {"motion_corrected_recording_id": corrected_id}
        expected["motion_corrected_recording_id"] = {
            "id": corrected_id,
            "exists": bool(MotionCorrectedRecordingSelection & corrected_key),
            "computed_exists": bool(MotionCorrectedRecording & corrected_key),
        }
    return expected


def _resource_notes(
    display_waveform_params_name: str,
    *,
    n_channels: int,
    sampling_rate_hz: "float | None",
    effective_config: "dict | None",
) -> list[str]:
    """Known allocations / scratch / worker settings, from tracked recipes.

    Pre-sort the unit count is unknown, so the waveform bound is stated per
    unit; multiply by the expected unit count for the buffer size. Channels
    per unit are bounded by the sort group's channel count (radius / best
    channels sparsity reduces it; dense uses all).
    """
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_cache_root,
    )
    from spyglass.spikesorting.v2._params.analyzer_waveform import (
        SparsityParams,
    )
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters
    from spyglass.settings import temp_dir

    notes: list[str] = []
    rows = (
        AnalyzerWaveformParameters
        & {"waveform_params_name": display_waveform_params_name}
    ).fetch(as_dict=True)
    if len(rows) == 1 and sampling_rate_hz:
        params = dict(rows[0]["params"])
        sparsity = SparsityParams.model_validate(params["sparsity"])
        n_samples = int(
            round(
                (float(params["ms_before"]) + float(params["ms_after"]))
                * sampling_rate_hz
                / 1000.0
            )
        )
        if sparsity.method == "best_channels":
            channels = min(int(sparsity.num_channels), int(n_channels))
            channel_note = f"{channels} channels (best_channels)"
        elif sparsity.method == "radius":
            channels = int(n_channels)
            channel_note = (
                f"<= {channels} channels (radius {sparsity.radius_um:g} um; "
                "fewer on wide groups)"
            )
        else:
            channels = int(n_channels)
            channel_note = f"{channels} channels (dense)"
        per_unit = int(params["max_spikes_per_unit"]) * n_samples * channels * 4
        notes.append(
            "display analyzer waveforms: up to "
            f"{int(params['max_spikes_per_unit'])} spikes/unit x {n_samples} "
            f"samples x {channel_note} x 4 bytes = up to "
            f"{per_unit / 1024**2:.1f} MiB per unit on disk (memmapped; "
            "extraction peak ~1.5x the total, loads are lazy)."
        )
    notes.append(
        f"analyzer cache root: {analyzer_cache_root()} (display analyzer, plus "
        "a whitened metric analyzer when PC metrics are requested, plus one "
        "per-generation cache per reviewed merged curation)."
    )
    notes.append(
        f"sorter scratch: a per-sort temporary directory under {temp_dir}, "
        "removed when the sort finishes; the preprocessed Recording is a "
        "separate NWB artifact."
    )
    if effective_config:
        jobs = effective_config.get("job_kwargs") or {}
        notes.append(
            "effective worker/chunk settings (sort + analyzer): "
            f"n_jobs={jobs.get('n_jobs', 1)}, "
            f"chunk_duration={jobs.get('chunk_duration', '1s')!r}, "
            f"random_seed={effective_config.get('random_seed')}, "
            f"external_whiten={effective_config.get('external_whiten')}."
        )
    return notes
