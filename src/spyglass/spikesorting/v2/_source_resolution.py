"""Resolve which source a sort came from and which traces it reads.

A sort answers "what did it run on?" in two separate ways:

- :class:`SourceLineage` is the sort's original source. It is the
  ``Recording`` or ``ConcatenatedRecording`` selected on ``SortingSelection``,
  plus the artifact detection pinned to the sort. Metadata consumers (anchor
  NWB, brain regions, curation routing) read lineage.
- :class:`EffectiveTraces` is the persisted artifact that holds the traces
  every trace consumer must read: the sorter input, analyzer builds and
  rebuilds, metric curation, the recompute audit, and the curation recording
  accessor. It also says whether the consumer must still apply the sort's
  artifact mask.

The effective traces are the lineage source itself, or a derived artifact:
a ``MotionCorrectedRecording`` (kind ``"motion_corrected_recording"``), which
is persisted already masked. Keeping the two apart lets metadata consumers
read the original source without inferring it from the derived artifact.

``SortingSelection.resolve_effective_source`` performs the DB reads and builds
the result with :func:`effective_source_from_base` or
:func:`effective_source_from_correction`; :func:`correction_lineage_mismatch`
checks that a corrected recording was made from the sort's own source and mask,
and :func:`sorting_parts_mismatch` that the parts still give the stored
``sorting_id``.
:func:`load_effective_recording` opens the traces, and
:func:`read_effective_recording` does the same from an already-resolved path;
:func:`read_persisted_traces` opens them as persisted, with no mask at load.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import. SpikeInterface and the NWB reader are imported lazily.
:func:`load_effective_recording` resolves the file path through
``AnalysisNwbfile.get_abs_path``, which reads the ``AnalysisNwbfile`` table;
:func:`read_effective_recording` and :func:`read_persisted_traces` touch no
DB.
"""

from __future__ import annotations

import uuid
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

    A sort may read a motion-corrected recording only if its motion was
    estimated on the sort's own source (same kind and key) under the same
    artifact mask (both ``None`` for a concatenated recording, which carries
    its member masks).

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

    ``sorting_id`` is derived from the source, the artifact detection, the
    motion-corrected recording and the sorter row when the selection is
    inserted. Recomputing it from the parts present now detects a part
    inserted or deleted around ``SortingSelection.insert_selection``, which
    would otherwise change the traces a sort's consumers read without
    changing the sort.

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
    from spyglass.spikesorting.v2._selection_identity import (
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

    Every consumer of a motion-corrected recording must see the channels the
    correction wrote, in order, at the positions it recorded. With
    ``border_mode="remove_channels"`` that is a subset of the source's
    channels; nothing may pad, reorder or drop channels to restore the
    source's set.

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
    if len(np.unique(np.round(positions, 6), axis=0)) != len(positions):
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

    Opens the file with ``_recording_nwb.open_persisted_traces`` (the stored
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
    from spyglass.spikesorting.v2._recording_nwb import open_persisted_traces

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
        from spyglass.spikesorting.v2._sorting_artifact_mask import (
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

    A single-recording source's cache comes back unmasked (a consumer that
    needs the mask applies it itself, e.g. through ``artifact_frame_ranges``),
    while a concat artifact keeps its member masks and a motion-corrected
    artifact keeps the sort's mask, both of which are written into the file.
    No DB access.

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


def load_effective_recording(
    traces: EffectiveTraces,
    *,
    artifact_valid_times: np.ndarray | None = None,
    artifact_detection_id: uuid.UUID | None = None,
    recording_id: uuid.UUID | None = None,
) -> si.BaseRecording:
    """Open the effective traces by analysis file name, masking if needed.

    Resolves ``traces.row["analysis_file_name"]`` through
    ``AnalysisNwbfile.get_abs_path`` and delegates to
    :func:`read_effective_recording`. The file must already exist; the
    owning table's self-heal (``SortingSelection.ensure_effective_traces``)
    is the caller's responsibility.

    Parameters
    ----------
    traces : EffectiveTraces
        The resolved traces artifact.
    artifact_valid_times : np.ndarray, optional
        Artifact-removed valid times, shape ``(n_intervals, 2)`` in seconds;
        see :func:`read_effective_recording`.
    artifact_detection_id : uuid.UUID, optional
        Passed to the mask for error messages.
    recording_id : uuid.UUID, optional
        Passed to the mask for error messages.

    Returns
    -------
    si.BaseRecording
        The traces, silenced over the artifact periods when masking applies.
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    return read_effective_recording(
        AnalysisNwbfile.get_abs_path(traces.row["analysis_file_name"]),
        traces,
        artifact_valid_times=artifact_valid_times,
        artifact_detection_id=artifact_detection_id,
        recording_id=recording_id,
    )
