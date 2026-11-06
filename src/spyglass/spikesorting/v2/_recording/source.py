"""Resolve which source a sort came from and which traces it reads.

- :class:`SourceLineage`: the sort's original ``Recording`` or
  ``ConcatenatedRecording`` plus its pinned artifact detection. Metadata
  consumers read this.
- :class:`EffectiveTraces`: the persisted artifact every trace consumer
  reads, either the lineage source or a ``MotionCorrectedRecording``, and
  whether the artifact mask must still be applied.

DB-free; SpikeInterface and the NWB reader are imported lazily.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    import numpy as np
    import spikeinterface as si

BaseTraceKind = Literal["recording", "concatenated_recording"]
TraceKind = Literal[
    "recording", "concatenated_recording", "motion_corrected_recording"
]


class SourceLineage(NamedTuple):
    """The sort's original source; it does not change with the traces read.

    Attributes
    ----------
    kind : {"recording", "concatenated_recording"}
        Which ``SortingSelection`` source part backs the sort.
    key : dict
        The source row's primary key: ``{"recording_id": ...}`` or
        ``{"concat_recording_id": ...}``.
    artifact_detection_id : uuid.UUID or None
        The artifact detection pinned on the sort's ``ArtifactDetectionSource``
        part, or ``None``. Only a single-recording sort pins one; a concat
        source owns its member masks upstream.
    """

    kind: BaseTraceKind
    key: dict
    artifact_detection_id: uuid.UUID | None


class EffectiveTraces(NamedTuple):
    """The persisted artifact holding the traces a sort's consumers read.

    Attributes
    ----------
    kind : {"recording", "concatenated_recording", "motion_corrected_recording"}
        The table that owns the traces artifact.
    key : dict
        That table's primary key for the artifact row.
    row : dict
        The fetched artifact row; carries at least ``analysis_file_name`` and
        ``electrical_series_path``.
    apply_artifact_mask : bool
        ``True`` only when a consumer must still silence the sort's artifact
        periods after loading (a single-recording source with a pinned
        detection). A concat artifact already has its member masks written
        in, and a motion-corrected artifact is persisted masked.
    """

    kind: TraceKind
    key: dict
    row: dict
    apply_artifact_mask: bool


class EffectiveSource(NamedTuple):
    """A sort's lineage together with the traces its consumers read."""

    lineage: SourceLineage
    traces: EffectiveTraces


def effective_source_from_base(
    lineage: SourceLineage, row: dict
) -> EffectiveSource:
    """Build the effective source for a sort that reads its lineage source.

    Parameters
    ----------
    lineage : SourceLineage
        The sort's resolved lineage.
    row : dict
        The fetched lineage source row (``Recording`` or
        ``ConcatenatedRecording``).

    Returns
    -------
    EffectiveSource
        ``traces`` names the lineage source itself. The mask is applied at load
        only for a single-recording source with a pinned detection; a concat
        artifact already carries its member masks, so a detection id on a
        concat lineage never triggers a second mask.
    """
    traces = EffectiveTraces(
        kind=lineage.kind,
        key=lineage.key,
        row=row,
        apply_artifact_mask=(
            lineage.kind == "recording"
            and lineage.artifact_detection_id is not None
        ),
    )
    return EffectiveSource(lineage=lineage, traces=traces)


def effective_source_from_correction(
    lineage: SourceLineage, key: dict, row: dict
) -> EffectiveSource:
    """Build the effective source for a sort that reads a corrected recording.

    Parameters
    ----------
    lineage : SourceLineage
        The sort's resolved lineage (unchanged by the correction).
    key : dict
        ``{"motion_corrected_recording_id": ...}``.
    row : dict
        The fetched ``MotionCorrectedRecording`` row.

    Returns
    -------
    EffectiveSource
        ``traces`` names the corrected recording, never masked at load: it is
        persisted with the sort's mask already applied.
    """
    traces = EffectiveTraces(
        kind="motion_corrected_recording",
        key=key,
        row=row,
        apply_artifact_mask=False,
    )
    return EffectiveSource(lineage=lineage, traces=traces)


def correction_lineage_mismatch(
    sort_lineage: SourceLineage,
    correction_lineage: SourceLineage,
    *,
    consumer: str = "sort",
) -> list[str]:
    """Describe how a corrected recording's source differs from a sort's.

    The correction must come from the sort's own source under the same
    artifact mask.

    Parameters
    ----------
    sort_lineage : SourceLineage
        The sort's source and pinned artifact detection.
    correction_lineage : SourceLineage
        The source and artifact detection of the corrected recording's
        motion estimate.
    consumer : str, optional
        What ``sort_lineage`` belongs to, named in each description
        (``"the sort's ..."`` by default; ``"run"`` for a pipeline run that
        applies a saved estimate before any sort exists).

    Returns
    -------
    list of str
        One description per differing field; empty when they agree.
    """

    def normalized(lineage: SourceLineage) -> tuple:
        detection = lineage.artifact_detection_id
        return (
            lineage.kind,
            {
                name: uuid.UUID(str(value))
                for name, value in lineage.key.items()
            },
            None if detection is None else uuid.UUID(str(detection)),
        )

    sort_kind, sort_key, sort_detection = normalized(sort_lineage)
    kind, key, detection = normalized(correction_lineage)
    mismatches = []
    if (kind, key) != (sort_kind, sort_key):
        mismatches.append(
            f"source {kind} {key} != the {consumer}'s {sort_kind} {sort_key}"
        )
    if detection != sort_detection:
        mismatches.append(
            f"artifact_detection_id {detection} != the {consumer}'s "
            f"{sort_detection}"
        )
    return mismatches


def sorting_parts_mismatch(
    sorting_id,
    lineage: SourceLineage,
    *,
    sorter: str,
    sorter_params_name: str,
    motion_corrected_recording_id,
) -> str | None:
    """Describe why a sort's current parts do not give its ``sorting_id``.

    Recomputing ``sorting_id`` from the current parts detects a part
    inserted or deleted outside ``SortingSelection.insert_selection``.

    Parameters
    ----------
    sorting_id : uuid.UUID or str
        The stored ``SortingSelection`` primary key.
    lineage : SourceLineage
        The source and artifact detection read from the selection's parts.
    sorter : str
        The selection's sorter.
    sorter_params_name : str
        The selection's ``SorterParameters`` name.
    motion_corrected_recording_id : uuid.UUID, str or None
        The ``MotionCorrectionSource`` part's corrected recording, or ``None``.

    Returns
    -------
    str or None
        ``None`` when the parts give ``sorting_id``; otherwise a description.
    """
    from spyglass.spikesorting.v2._core.selection_identity import (
        deterministic_id,
        sorting_identity_payload,
    )

    if lineage.kind == "concatenated_recording":
        if lineage.artifact_detection_id is not None:
            return (
                "a concatenated-recording source with artifact_detection_id "
                f"{lineage.artifact_detection_id}, which no selection can have"
            )
        source = {"concat_recording_id": lineage.key["concat_recording_id"]}
    else:
        source = {
            "recording_id": lineage.key["recording_id"],
            "artifact_detection_id": lineage.artifact_detection_id,
        }
    expected = deterministic_id(
        "sorting",
        sorting_identity_payload(
            sorter=sorter,
            sorter_params_name=sorter_params_name,
            motion_corrected_recording_id=motion_corrected_recording_id,
            **source,
        ),
    )
    if expected == uuid.UUID(str(sorting_id)):
        return None
    return (
        f"source {lineage.kind} {lineage.key}, artifact_detection_id "
        f"{lineage.artifact_detection_id}, motion_corrected_recording_id "
        f"{motion_corrected_recording_id} give sorting_id {expected}"
    )


def check_corrected_channel_map(
    recording: si.BaseRecording, traces: EffectiveTraces
) -> None:
    """Check a loaded corrected recording against its row's channel map.

    Channels must match the row exactly, in order; with
    ``border_mode="remove_channels"`` they are a subset of the source's.

    Parameters
    ----------
    recording : si.BaseRecording
        The corrected recording as read from its NWB artifact.
    traces : EffectiveTraces
        Its resolved traces; ``row`` is the ``MotionCorrectedRecording`` row
        (``channel_ids``, ``channel_locations`` ``(n_channels, 2)`` in um).

    Raises
    ------
    ValueError
        If the loaded channel ids differ from the row's (values or order), or
        the loaded contact positions are not finite, not distinct, or differ
        from the row's.
    """
    import numpy as np

    from spyglass.spikesorting.v2._recording.geometry import _all_rows_distinct

    row = traces.row
    loaded_ids = [str(c) for c in recording.channel_ids]
    stored_ids = [str(c) for c in np.asarray(row["channel_ids"]).tolist()]
    if loaded_ids != stored_ids:
        raise ValueError(
            f"{traces.kind} {traces.key}: the artifact's channels "
            f"{loaded_ids} differ from the row's channel_ids {stored_ids}."
        )
    positions = np.asarray(recording.get_channel_locations(), dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError(
            f"{traces.kind} {traces.key}: contact positions must be finite; "
            f"got {positions.tolist()}."
        )
    if not _all_rows_distinct(positions):
        raise ValueError(
            f"{traces.kind} {traces.key}: two or more contacts share a "
            f"position ({positions.tolist()})."
        )
    stored = np.asarray(row["channel_locations"], dtype=float)
    if not np.array_equal(positions, stored):
        raise ValueError(
            f"{traces.kind} {traces.key}: the artifact's contact positions "
            f"{positions.tolist()} differ from the row's channel_locations "
            f"{stored.tolist()}."
        )


def read_effective_recording(
    abs_path: str,
    traces: EffectiveTraces,
    *,
    artifact_valid_times: np.ndarray | None = None,
    artifact_detection_id: uuid.UUID | None = None,
    recording_id: uuid.UUID | None = None,
) -> si.BaseRecording:
    """Open the effective traces from a resolved file path, masking if needed.

    Opens the file with ``_storage.nwb.open_persisted_traces`` (the stored
    ``electrical_series_path``, annotated ``is_filtered=True``). The artifact
    mask is applied exactly when ``traces.apply_artifact_mask`` is set.

    Parameters
    ----------
    abs_path : str
        Absolute path of the analysis NWB named by
        ``traces.row["analysis_file_name"]``.
    traces : EffectiveTraces
        The resolved traces artifact.
    artifact_valid_times : np.ndarray, optional
        Artifact-removed valid times, shape ``(n_intervals, 2)`` in seconds.
        Required when ``traces.apply_artifact_mask`` is ``True`` and rejected
        otherwise.
    artifact_detection_id : uuid.UUID, optional
        Passed to the mask for error messages.
    recording_id : uuid.UUID, optional
        Passed to the mask for error messages.

    Returns
    -------
    si.BaseRecording
        The traces, silenced over the artifact periods when masking applies.

    Raises
    ------
    ValueError
        If ``artifact_valid_times`` is missing while the mask applies, or given
        while it does not, or if a motion-corrected artifact (persisted
        masked) is asked to be masked again.
    """
    from spyglass.spikesorting.v2._storage.nwb import open_persisted_traces

    if traces.kind == "motion_corrected_recording" and (
        traces.apply_artifact_mask
    ):
        raise ValueError(
            f"{traces.kind} {traces.key} is persisted with its mask applied; "
            "it must not be artifact-masked again at load."
        )

    if traces.apply_artifact_mask and artifact_valid_times is None:
        raise ValueError(
            f"{traces.kind} {traces.key} must be artifact-masked at load, but "
            "no artifact_valid_times were supplied."
        )
    if not traces.apply_artifact_mask and artifact_valid_times is not None:
        raise ValueError(
            f"{traces.kind} {traces.key} is not artifact-masked at load, but "
            "artifact_valid_times were supplied."
        )
    recording = open_persisted_traces(
        abs_path, traces.row["electrical_series_path"]
    )
    if traces.kind == "motion_corrected_recording":
        check_corrected_channel_map(recording, traces)
    if traces.apply_artifact_mask:
        from spyglass.spikesorting.v2._sorting.artifact_mask import (
            apply_artifact_mask,
        )

        recording = apply_artifact_mask(
            recording,
            artifact_valid_times,
            artifact_detection_id=artifact_detection_id,
            recording_id=recording_id,
        )
    return recording


def read_persisted_traces(
    abs_path: str, traces: EffectiveTraces
) -> si.BaseRecording:
    """Open the effective traces as persisted: no artifact mask at load.

    A single-recording cache comes back unmasked; concat and
    motion-corrected artifacts have their masks written in. No DB access.

    Parameters
    ----------
    abs_path : str
        Absolute path of the analysis NWB named by
        ``traces.row["analysis_file_name"]``.
    traces : EffectiveTraces
        The resolved traces artifact.

    Returns
    -------
    si.BaseRecording
        The persisted traces, annotated ``is_filtered=True``.
    """
    return read_effective_recording(
        abs_path, traces._replace(apply_artifact_mask=False)
    )


@dataclass(frozen=True)
class SourceResolution:
    """Result of ``SortingSelection.resolve_source(key)``.

    ``key`` is ``{"recording_id": ...}`` or ``{"concat_recording_id": ...}``.
    """

    kind: Literal["recording", "concatenated_recording"]
    key: dict
