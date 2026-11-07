"""Pure steps of the motion tables' ``make_compute``.

They read only their arguments and the source NWB file; no database or
staging. Heavy dependencies are imported inside the functions.
"""

from __future__ import annotations

import numpy as np

from spyglass.spikesorting.v2._motion import estimation as _motion
from spyglass.spikesorting.v2._sorting.artifact_mask import (
    artifact_frame_ranges,
    continuity_from_timestamps,
    statistics_spans,
)


def stale_fields(checks) -> list[str]:
    """Describe each ``(name, current, selected)`` check whose values differ.

    Parameters
    ----------
    checks : iterable of (str, object, object)
        A field's name, its value now and its value on the selection.

    Returns
    -------
    list[str]
        One ``"<name> <current> != selected <selected>"`` per stale field.
    """
    return [
        f"{name} {now!r} != selected {then!r}"
        for name, now, then in checks
        if now != then
    ]


def stale_corrected_selection_fields(
    selection: dict,
    resolved_interpolation_params: dict,
    *,
    check_spikeinterface_version: bool,
) -> list[str]:
    """Describe what changed since a corrected recording was selected.

    Parameters
    ----------
    selection : dict
        The ``MotionCorrectedRecordingSelection`` row.
    resolved_interpolation_params : dict
        Its ``MotionInterpolationParameters`` row's ``params`` blob, resolved
        by ``_motion.resolve_interpolation_params``.
    check_spikeinterface_version : bool
        Also compare the installed SpikeInterface version.

    Returns
    -------
    list[str]
        One description per stale field (:func:`stale_fields`); empty when
        the selection is current.
    """
    import spikeinterface as si

    checks = [
        (
            "resolved interpolation hash",
            _motion.resolved_params_hash(resolved_interpolation_params),
            selection["resolved_params_hash"],
        )
    ]
    if check_spikeinterface_version:
        checks.append(
            (
                "SpikeInterface version",
                si.__version__,
                selection["spikeinterface_version"],
            )
        )
    checks.append(
        (
            "motion interpolation algorithm version",
            _motion.MOTION_INTERPOLATION_ALGORITHM_VERSION,
            selection["motion_interpolation_algorithm_version"],
        )
    )
    return stale_fields(checks)


def stale_estimate_selection_fields(
    selection: dict, resolved_hash: str
) -> list[str]:
    """Describe what changed since a motion estimate was selected.

    Parameters
    ----------
    selection : dict
        The ``MotionEstimateSelection`` row.
    resolved_hash : str
        ``_motion.resolved_params_hash`` of its recipe resolved now.

    Returns
    -------
    list[str]
        One description per stale field (:func:`stale_fields`): the
        resolved configuration hash, the SpikeInterface version and the
        motion algorithm version; empty when the selection is current.
    """
    import spikeinterface as si

    return stale_fields(
        [
            (
                "resolved configuration hash",
                resolved_hash,
                selection["resolved_params_hash"],
            ),
            (
                "SpikeInterface version",
                si.__version__,
                selection["spikeinterface_version"],
            ),
            (
                "motion algorithm version",
                _motion.MOTION_ALGORITHM_VERSION,
                selection["motion_algorithm_version"],
            ),
        ]
    )


def resolve_estimation_recipe(
    key, selection: dict, params: dict
) -> tuple[dict, str]:
    """Resolve an estimate's recipe, refusing a stale selection.

    Parameters
    ----------
    key : dict
        The ``MotionEstimate`` key, for the message.
    selection : dict
        The ``MotionEstimateSelection`` row.
    params : dict
        Its ``MotionEstimationParameters`` row's ``params`` blob.

    Returns
    -------
    resolved : dict
        The resolved estimation configuration.
    resolved_hash : str
        ``_motion.resolved_params_hash`` of ``resolved``.

    Raises
    ------
    ValueError
        If the selection is stale (:func:`stale_estimate_selection_fields`).
    """
    resolved = _motion.resolve_estimation_params(params)
    resolved_hash = _motion.resolved_params_hash(resolved)
    stale = stale_estimate_selection_fields(selection, resolved_hash)
    if stale:
        raise ValueError(
            f"MotionEstimate {key}: the selection is stale ({'; '.join(stale)}). "
            "Recreate it with MotionEstimateSelection.insert_selection, "
            "which derives a new motion_estimate_id."
        )
    return resolved, resolved_hash


def resolve_interpolation_recipe(
    key,
    selection: dict,
    interpolation_params: dict,
    *,
    check_spikeinterface_version: bool,
) -> dict:
    """Resolve a corrected recording's recipe, refusing a stale selection.

    Parameters
    ----------
    key : dict
        The ``MotionCorrectedRecording`` key, for the message.
    selection : dict
        The ``MotionCorrectedRecordingSelection`` row.
    interpolation_params : dict
        Its ``MotionInterpolationParameters`` row's ``params`` blob.
    check_spikeinterface_version : bool
        Also compare the installed SpikeInterface version.

    Returns
    -------
    dict
        The resolved interpolation configuration.

    Raises
    ------
    ValueError
        If the selection is stale (:func:`stale_corrected_selection_fields`).
    """
    resolved = _motion.resolve_interpolation_params(interpolation_params)
    stale = stale_corrected_selection_fields(
        selection,
        resolved,
        check_spikeinterface_version=check_spikeinterface_version,
    )
    if stale:
        raise ValueError(
            f"MotionCorrectedRecording {key}: the selection is stale "
            f"({'; '.join(stale)}). Recreate it with "
            "MotionCorrectedRecordingSelection.insert_selection."
        )
    return resolved


def estimation_spans(
    recording, lineage, source_row: dict, artifact_valid_times, n_samples: int
) -> tuple:
    """The continuity and statistics spans a source is estimated in.

    For a single recording, the continuity spans and each span's first and
    last timestamp come from the persisted timestamps, and the statistics
    spans from the artifact ranges, as the sort stage computes them; for a
    concatenation, all of them come from its row.

    Parameters
    ----------
    recording : si.BaseRecording
        The source traces.
    lineage : SourceLineage
        The estimate's source kind, key and pinned artifact detection.
    source_row : dict
        The ``Recording`` / ``ConcatenatedRecording`` row.
    artifact_valid_times : numpy.ndarray or None
        ``(n_intervals, 2)`` artifact-removed valid times in seconds, for a
        masked single-recording source.
    n_samples : int
        Frames of ``recording``.

    Returns
    -------
    continuity : list of tuple of int
        Half-open frame ranges of uninterrupted acquisition.
    continuity_start_s, continuity_end_s : sequence of float
        First and last timestamp of each span, in seconds (lists for a single
        recording, the stored arrays for a concatenation).
    statistics : list of tuple of int
        Half-open frame ranges of the valid samples.
    """
    if lineage.kind == "recording":
        continuity, continuity_start_s, continuity_end_s = (
            continuity_from_timestamps(recording)
        )
        excluded = []
        if artifact_valid_times is not None:
            excluded = artifact_frame_ranges(
                recording,
                artifact_valid_times,
                artifact_detection_id=lineage.artifact_detection_id,
                recording_id=lineage.key["recording_id"],
            )
        statistics = statistics_spans(n_samples, excluded, continuity)
    else:
        continuity = _motion.normalize_spans(source_row["continuity_spans"])
        continuity_start_s = source_row["continuity_start_s"]
        continuity_end_s = source_row["continuity_end_s"]
        statistics = _motion.normalize_spans(source_row["statistics_spans"])

    return continuity, continuity_start_s, continuity_end_s, statistics


def assert_source_matches_estimate(
    key, source, estimate: dict, n_samples: int, sampling_frequency: float
) -> None:
    """Refuse a source whose frames, rate or channels differ from an estimate's.

    Parameters
    ----------
    key : dict
        The ``MotionCorrectedRecording`` key, for the message.
    source : si.BaseRecording
        The source traces.
    estimate : dict
        The saved estimate's application fields.
    n_samples : int
        Frames of ``source``.
    sampling_frequency : float
        Sampling frequency of ``source`` in Hz.

    Raises
    ------
    ValueError
        If any of them differs: the estimate does not describe the traces.
    """
    mismatched = [
        name
        for name, now, then in (
            ("n_samples", n_samples, int(estimate["n_samples"])),
            (
                "sampling_frequency",
                sampling_frequency,
                float(estimate["sampling_frequency"]),
            ),
            (
                "channel_ids",
                source.channel_ids.tolist(),
                np.asarray(estimate["channel_ids"]).tolist(),
            ),
        )
        if now != then
    ]
    if mismatched:
        raise ValueError(
            f"MotionCorrectedRecording {key}: the source's {mismatched} "
            "differ from the saved estimate's; the estimate does not "
            "describe these traces."
        )


def assert_geometry_matches_estimate(key, source, estimate: dict) -> None:
    """Refuse a source whose contact positions differ from an estimate's.

    ``source`` must already carry the flattened positions the estimate was
    computed on.

    Raises
    ------
    ValueError
        If the positions differ: the estimate does not describe the
        geometry.
    """
    source_locations = np.asarray(
        source.get_channel_locations(), dtype=np.float64
    )
    estimate_locations = np.asarray(
        estimate["channel_locations"], dtype=np.float64
    )
    if not np.array_equal(source_locations, estimate_locations):
        raise ValueError(
            f"MotionCorrectedRecording {key}: the source's contact "
            f"positions {source_locations.tolist()} differ from the saved "
            f"estimate's {estimate_locations.tolist()}; the estimate does "
            "not describe this geometry."
        )


def apply_saved_motion(
    key, source, estimate: dict, clock, resolved: dict, n_samples: int
) -> tuple:
    """Apply a saved estimate to its source on the estimation clock.

    Parameters
    ----------
    key : dict
        The ``MotionCorrectedRecording`` key, for the message.
    source : si.BaseRecording
        The source traces, with the flattened geometry the estimate used.
    estimate : dict
        The saved estimate's application fields.
    clock : _motion.EstimationClock
        The estimate's estimation clock.
    resolved : dict
        The resolved interpolation configuration.
    n_samples : int
        Frames of ``source``.

    Returns
    -------
    applied : _motion.AppliedMotion
        The applied correction.
    statistics : numpy.ndarray
        ``(n, 2)`` int64 statistics spans the correction silenced outside.

    Raises
    ------
    ValueError
        If the interpolation changed the sample count.
    """
    statistics = np.asarray(
        estimate["statistics_spans"], dtype=np.int64
    ).reshape(-1, 2)
    applied = _motion.apply_motion_on_estimation_clock(
        source,
        _motion.motion_from_storage_dict(estimate["motion"]),
        clock=clock,
        statistics_spans=statistics,
        resolved_interpolation=resolved,
    )
    corrected = applied.recording
    if int(corrected.get_num_samples()) != n_samples:
        raise ValueError(
            f"MotionCorrectedRecording {key}: interpolation changed the "
            f"sample count ({corrected.get_num_samples()} != {n_samples})."
        )
    return applied, statistics


def corrected_filtering_description(
    source_filtering: str, motion_estimate_id, resolved: dict
) -> str:
    """The ``filtering`` text of a corrected recording's series.

    Parameters
    ----------
    source_filtering : str
        The source series' ``filtering`` attribute.
    motion_estimate_id : uuid.UUID or str
        The applied estimate.
    resolved : dict
        The resolved interpolation configuration.

    Returns
    -------
    str
    """
    interpolation_text = ", ".join(
        f"{name}={resolved[name]}" for name in sorted(resolved)
    )
    return (
        f"{source_filtering}; motion corrected with saved estimate "
        f"{motion_estimate_id} ({interpolation_text}); "
        "frames outside the statistics spans silenced; unwhitened"
    )


def motion_correction_provenance_tables(
    key,
    selection: dict,
    resolved: dict,
    source_content_hash,
    applied,
    source_kind: str,
    source_key: dict,
    statistics: np.ndarray,
    clock,
    source_path: str,
) -> list:
    """The provenance tables a corrected recording's NWB carries.

    The correction's ids, recipe and source, the statistics spans, each
    continuity span with its real start and end times and its start on the
    estimation clock, and, for a concatenation, the concatenation's member
    back-map copied from the source artifact.

    Parameters
    ----------
    key : dict
        The ``MotionCorrectedRecording`` key.
    selection : dict
        The ``MotionCorrectedRecordingSelection`` row.
    resolved : dict
        The resolved interpolation configuration.
    source_content_hash : str
        ``content_hash`` of the source trace artifact.
    applied : _motion.AppliedMotion
        The applied correction.
    source_kind : str
        ``"recording"`` or ``"concatenated_recording"``.
    source_key : dict
        The source row's primary key.
    statistics : numpy.ndarray
        ``(n, 2)`` int64 statistics spans.
    clock : _motion.EstimationClock
        The estimate's estimation clock.
    source_path : str
        Absolute path of the source's analysis NWB.

    Returns
    -------
    list
        ``DynamicTable`` objects for ``write_nwb_artifact``'s
        ``provenance_tables``.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2._storage.provenance import (
        CONCAT_MEMBER_COLUMNS,
        CONCAT_MEMBERS,
        MOTION_CONTINUITY_SPAN_COLUMNS,
        MOTION_CONTINUITY_SPANS,
        MOTION_CORRECTION_PROVENANCE,
        build_long_provenance_table,
        build_provenance_table,
        read_long_provenance,
    )

    provenance_tables = [
        build_provenance_table(
            MOTION_CORRECTION_PROVENANCE,
            {
                "motion_corrected_recording_id": str(
                    key["motion_corrected_recording_id"]
                ),
                "motion_estimate_id": str(selection["motion_estimate_id"]),
                "motion_interpolation_params_name": selection[
                    "motion_interpolation_params_name"
                ],
                "interpolation": resolved,
                "motion_interpolation_algorithm_version": int(
                    selection["motion_interpolation_algorithm_version"]
                ),
                "source_content_hash": str(source_content_hash),
                "removed_channel_ids": applied.removed_channel_ids,
                "spikeinterface_version": si.__version__,
                "source_kind": source_kind,
                "source_key": {
                    name: str(value) for name, value in source_key.items()
                },
                "statistics_spans": statistics.tolist(),
                "estimation_clock_sampling_frequency": (
                    clock.sampling_frequency
                ),
            },
        ),
        build_long_provenance_table(
            MOTION_CONTINUITY_SPANS,
            [
                {
                    "span_index": index,
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "source_start_s": float(source_start),
                    "source_end_s": float(source_end),
                    "estimation_start_s": float(estimation_start),
                }
                for index, (
                    (start, end),
                    source_start,
                    source_end,
                    estimation_start,
                ) in enumerate(
                    zip(
                        clock.spans,
                        clock.source_start_s,
                        clock.source_end_s,
                        clock.estimation_start_s,
                        strict=True,
                    )
                )
            ],
            MOTION_CONTINUITY_SPAN_COLUMNS,
        ),
    ]
    if source_kind == "concatenated_recording":
        provenance_tables.append(
            build_long_provenance_table(
                CONCAT_MEMBERS,
                read_long_provenance(source_path, CONCAT_MEMBERS),
                CONCAT_MEMBER_COLUMNS,
            )
        )
    return provenance_tables
