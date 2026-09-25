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

Today the effective traces are always the lineage source itself. Keeping the
two apart lets a derived trace artifact be selected later without metadata
consumers having to infer the original source from it.

``SortingSelection.resolve_effective_source`` performs the DB reads and builds
the result with :func:`effective_source_from_base`.
:func:`load_effective_recording` opens the traces, and
:func:`read_effective_recording` does the same from an already-resolved path.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import. SpikeInterface and the NWB reader are imported lazily.
:func:`load_effective_recording` resolves the file path through
``AnalysisNwbfile.get_abs_path``, which reads the ``AnalysisNwbfile`` table;
:func:`read_effective_recording` touches no DB.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    import numpy as np
    import spikeinterface as si

BaseTraceKind = Literal["recording", "concatenated_recording"]


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
    kind : {"recording", "concatenated_recording"}
        The table that owns the traces artifact.
    key : dict
        That table's primary key for the artifact row.
    row : dict
        The fetched artifact row; carries at least ``analysis_file_name`` and
        ``electrical_series_path``.
    apply_artifact_mask : bool
        ``True`` only when a consumer must still silence the sort's artifact
        periods after loading (a single-recording source with a pinned
        detection). A concat artifact already has its member masks written in.
    """

    kind: BaseTraceKind
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


def read_effective_recording(
    abs_path: str,
    traces: EffectiveTraces,
    *,
    artifact_valid_times: np.ndarray | None = None,
    artifact_detection_id: uuid.UUID | None = None,
    recording_id: uuid.UUID | None = None,
) -> si.BaseRecording:
    """Open the effective traces from a resolved file path, masking if needed.

    Reads the stored ``electrical_series_path`` (authoritative, not an
    auto-detect hint) and annotates ``is_filtered=True``: the persisted traces
    are already bandpass-filtered and referenced, so a downstream SpikeInterface
    consumer must not filter them again. The artifact mask is applied exactly
    when ``traces.apply_artifact_mask`` is set.

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
        while it does not.
    """
    from spyglass.spikesorting.v2._recording_nwb import read_recording_nwb

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
    recording = read_recording_nwb(
        abs_path,
        electrical_series_path=traces.row["electrical_series_path"],
    )
    recording.annotate(is_filtered=True)
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
