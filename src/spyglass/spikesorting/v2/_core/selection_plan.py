"""Builders for the v2 selection insert plans.

The builders validate a request, compute the deterministic primary key, and
shape the rows to insert. The table's ``insert_selection`` runs the DB half.
A "plan" is the computed set of rows to insert.
"""

from __future__ import annotations

import uuid
from typing import Literal, NamedTuple

from spyglass.spikesorting.v2._core.selection_identity import (
    assert_supplied_id_matches,
    deterministic_id,
    recording_identity_payload,
    reject_unknown_fields,
    sorting_identity_payload,
)

#: Every field ``SortingSelection.insert_selection`` accepts. Others are
#: refused, since a misspelled optional key would select a different sort.
SORTING_SELECTION_FIELDS = frozenset(
    {
        "recording_id",
        "concat_recording_id",
        "sorter",
        "sorter_params_name",
        "artifact_detection_id",
        "motion_corrected_recording_id",
        "sorting_id",
    }
)


class RecordingSelectionPlan(NamedTuple):
    """The row + restriction a ``RecordingSelection`` insert needs.

    ``master_restriction`` is the logical identity; ``master_row`` adds the
    ``recording_id``.
    """

    recording_id: uuid.UUID
    master_restriction: dict
    master_row: dict


class SortingSelectionPlan(NamedTuple):
    """The rows + restrictions a ``SortingSelection`` insert needs.

    - Exactly one of ``recording_source_row`` / ``concat_source_row`` is set,
      matching ``source_kind``.
    - ``artifact_detection_id`` and ``motion_corrected_recording_id`` are
      normalized UUIDs or ``None``. ``insert_selection`` builds the
      ``ArtifactDetectionSource`` row, which needs a DB-side merge id.
    - ``source_kind`` ``"concat"`` corresponds to
      ``SourceResolution.kind`` ``"concatenated_recording"``.
    """

    sorting_id: uuid.UUID
    source_kind: Literal["recording", "concat"]
    master_restriction: dict
    source_restriction: dict
    master_row: dict
    recording_source_row: dict | None
    concat_source_row: dict | None
    artifact_detection_id: uuid.UUID | None
    motion_corrected_recording_id: uuid.UUID | None


def build_recording_selection_plan(
    key: dict, *, recording_input_hash: str
) -> RecordingSelectionPlan:
    """Validate a ``RecordingSelection`` request and build its insert plan.

    The identity is the FK set plus ``recording_input_hash``
    (:func:`recording_input_hash`, resolved DB-side by the caller), so changed
    construction inputs mint a new recording.

    Parameters
    ----------
    key : dict
        The ``RecordingSelection`` FK fields and optional ``recording_id``.
    recording_input_hash : str
        Digest of the resolved construction inputs.

    Returns
    -------
    RecordingSelectionPlan

    Raises
    ------
    ValueError
        If a required identity field is missing, if ``key`` carries a field
        other than the identity fields and the optional ``recording_id``, or
        if an explicit ``recording_id`` does not equal the derived
        deterministic id.
    """
    identity_payload = recording_identity_payload(key)
    master_restriction = {
        **identity_payload,
        "recording_input_hash": recording_input_hash,
    }
    recording_id = deterministic_id("recording", master_restriction)
    assert_supplied_id_matches(
        key.get("recording_id"), recording_id, field="recording_id"
    )
    master_row = {**master_restriction, "recording_id": recording_id}
    return RecordingSelectionPlan(
        recording_id=recording_id,
        master_restriction=master_restriction,
        master_row=master_row,
    )


def build_sorting_selection_plan(key: dict) -> SortingSelectionPlan:
    """Validate a ``SortingSelection`` request and build its insert plan.

    ``key`` needs exactly one of ``recording_id`` or ``concat_recording_id``,
    plus ``sorter`` and ``sorter_params_name``. A concat source cannot carry
    ``artifact_detection_id`` because its member masks are frozen upstream.

    Raises
    ------
    ValueError
        If ``key`` carries a field outside :data:`SORTING_SELECTION_FIELDS`,
        if zero or both source keys are supplied, if ``sorter`` /
        ``sorter_params_name`` is missing, if a concat source is combined
        with an ``artifact_detection_id``, or if an explicit ``sorting_id``
        does not equal the derived deterministic id.
    """
    reject_unknown_fields(
        key,
        SORTING_SELECTION_FIELDS,
        caller="SortingSelection.insert_selection",
    )
    has_recording = "recording_id" in key
    has_concat = "concat_recording_id" in key
    if has_recording == has_concat:
        raise ValueError(
            "SortingSelection.insert_selection requires exactly one "
            "source key. Provide either recording_id (single-session) "
            "or concat_recording_id (concat). Got: "
            f"recording_id={'set' if has_recording else 'unset'}, "
            f"concat_recording_id={'set' if has_concat else 'unset'}."
        )

    for required in ("sorter", "sorter_params_name"):
        if required not in key:
            raise ValueError(
                f"SortingSelection.insert_selection requires "
                f"{required!r} in key."
            )

    master_restriction = {
        "sorter": key["sorter"],
        "sorter_params_name": key["sorter_params_name"],
    }
    artifact_detection_id = key.get("artifact_detection_id")
    # A str id would never match the stored UUID in find-existing.
    if artifact_detection_id is not None:
        artifact_detection_id = uuid.UUID(str(artifact_detection_id))
    motion_corrected_recording_id = key.get("motion_corrected_recording_id")
    if motion_corrected_recording_id is not None:
        motion_corrected_recording_id = uuid.UUID(
            str(motion_corrected_recording_id)
        )

    if has_concat and artifact_detection_id is not None:
        raise ValueError(
            "SortingSelection.insert_selection: a concat_recording_id "
            "source cannot supply an artifact_detection_id. Concat sorts "
            "own per-member masks through ConcatenatedRecordingSelection; "
            "supply the detections when selecting the concat source."
        )
    source_field = "concat_recording_id" if has_concat else "recording_id"
    source_restriction = {source_field: key[source_field]}
    identity = sorting_identity_payload(
        **source_restriction,
        sorter=key["sorter"],
        sorter_params_name=key["sorter_params_name"],
        artifact_detection_id=artifact_detection_id,
        motion_corrected_recording_id=motion_corrected_recording_id,
    )
    sorting_id = deterministic_id("sorting", identity)
    assert_supplied_id_matches(
        key.get("sorting_id"), sorting_id, field="sorting_id"
    )

    master_row = {**master_restriction, "sorting_id": sorting_id}
    source_row = {"sorting_id": sorting_id, **source_restriction}
    return SortingSelectionPlan(
        sorting_id=sorting_id,
        source_kind="concat" if has_concat else "recording",
        master_restriction=master_restriction,
        source_restriction=source_restriction,
        master_row=master_row,
        recording_source_row=None if has_concat else source_row,
        concat_source_row=source_row if has_concat else None,
        artifact_detection_id=artifact_detection_id,
        motion_corrected_recording_id=motion_corrected_recording_id,
    )
