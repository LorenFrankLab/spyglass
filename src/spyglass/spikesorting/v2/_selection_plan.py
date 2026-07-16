"""Pure builders for the v2 selection insert plans.

``RecordingSelection`` / ``SortingSelection`` /
``ArtifactDetectionSelection`` ``insert_selection`` are split into a pure
half (here) and a DB half (the orchestrator). The builders validate the
selection request, normalize ids, compute the deterministic content-
addressed primary key, and assemble the master + source part rows to
insert. The orchestrator then runs find-existing, the FK pre-checks, and
the atomic insert transaction (with duplicate-PK race recovery) on the
returned plan.

Isolating the pure half makes the input-validation + identity + row-shaping
logic unit-testable without a DataJoint transaction -- so the
misconfiguration paths (bad source combinations, supplied-id mismatch,
missing required keys, str-vs-UUID normalization) are cheap to pin.

Dependency-light: imports only :mod:`_selection_identity` (itself DB-free)
-- no v2 schema modules, no ``spyglass.common``.

Naming note: "plan" here (and in :mod:`_curation_plan`) is a domain noun -- the
computed set of rows to insert, like a query plan -- NOT a project-phase or
milestone reference.
"""

from __future__ import annotations

import uuid
from typing import Literal, NamedTuple

from spyglass.spikesorting.v2._selection_identity import (
    artifact_detection_identity_payload,
    assert_supplied_id_matches,
    deterministic_id,
    recording_identity_payload,
    sorting_identity_payload,
)


class RecordingSelectionPlan(NamedTuple):
    """The row + restriction a ``RecordingSelection`` insert needs.

    ``RecordingSelection`` has no source part: its logical identity is the
    full FK set (Raw, SortGroupV2, IntervalList, PreprocessingParameters,
    LabTeam), which doubles as ``master_restriction`` for find-existing.
    ``master_row`` is that FK set plus the content-addressed ``recording_id``.
    """

    recording_id: uuid.UUID
    master_restriction: dict
    master_row: dict


class SortingSelectionPlan(NamedTuple):
    """The rows + restrictions a ``SortingSelection`` insert needs.

    ``source_kind`` is ``"recording"`` or ``"concat"``; exactly one of
    ``recording_source_row`` / ``concat_source_row`` is non-``None`` and the
    other is ``None``, so the orchestrator inserts the matching source part
    (``RecordingSource`` xor ``ConcatenatedRecordingSource``) without
    re-deriving the source shape. ``source_restriction`` carries
    ``{"recording_id": ...}`` or ``{"concat_recording_id": ...}`` accordingly.

    ``artifact_source_row`` is ``None`` when no artifact-detection pass was
    requested (``artifact_detection_id`` absent) -- always the case for a
    concat source, which has no artifact pass; otherwise it is the
    ``ArtifactDetectionSource`` part row. ``artifact_detection_id`` is the
    normalized (``uuid.UUID``) value, threaded for find-existing and the
    artifact source part.

    Note: ``source_kind`` here is ``"concat"`` (the plan/identity vocabulary),
    which deliberately differs from ``utils.SourceResolution.kind``'s
    ``"concatenated_recording"`` (the post-insert resolution vocabulary); the
    two label the same source family at different layers.
    """

    sorting_id: uuid.UUID
    source_kind: Literal["recording", "concat"]
    master_restriction: dict
    source_restriction: dict
    master_row: dict
    recording_source_row: dict | None
    concat_source_row: dict | None
    artifact_source_row: dict | None
    artifact_detection_id: uuid.UUID | None


class ArtifactDetectionSelectionPlan(NamedTuple):
    """The rows + restrictions an ``ArtifactDetectionSelection`` insert needs.

    ``source_kind`` is ``"recording"`` or ``"shared_artifact_group"``; the
    orchestrator maps it to the matching source part table (the table class
    is a DB object and so stays out of this pure plan).
    """

    artifact_detection_id: uuid.UUID
    source_kind: Literal["recording", "shared_artifact_group"]
    master_restriction: dict
    source_restriction: dict
    master_row: dict
    source_row: dict


def build_recording_selection_plan(
    key: dict, *, recording_input_hash: str
) -> RecordingSelectionPlan:
    """Validate a ``RecordingSelection`` request and build its insert plan.

    The logical identity is the full FK set (Raw, SortGroupV2, IntervalList,
    PreprocessingParameters, LabTeam) PLUS ``recording_input_hash`` -- the
    content-address of the recording's RESOLVED construction inputs (electrode
    membership, reference, and the ``interpolate`` bad-channel set) that live
    outside the FK set but shape the recording's content (see
    :func:`recording_input_hash`). Folding it into the ``recording_id`` makes a
    changed input set mint a NEW recording rather than silently aliasing (or
    later failing to rebuild) an existing ``recording_id``. The caller resolves
    the hash DB-side (the resolution needs the live ``SortGroupElectrode`` /
    ``Electrode`` rows) and passes it here so this builder stays DB-free.

    FK-field validation (unknown-field rejection, required-field checks) is
    delegated to :func:`recording_identity_payload`, the single source of truth
    shared with ``preflight_v2_pipeline``; the hash is folded in identically at
    both sites so the two cannot derive different ids. The hash also enters
    ``master_restriction`` so find-existing is scoped to the specific input set
    -- two membership sets under one ``sort_group_id`` are distinct rows, not a
    false "non-deterministic bypass".

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

    Reads exactly one of ``recording_id`` (single-session) or
    ``concat_recording_id`` (concat) from ``key``; ``sorter`` and
    ``sorter_params_name`` are required. ``artifact_detection_id`` is
    optional on the recording path and, when present, normalized to a
    ``uuid.UUID`` so a caller-supplied ``str`` shares one identity with the
    stored value. A concat source must NOT carry an ``artifact_detection_id``:
    concat sorts reuse per-member ``Recording`` artifacts and have no
    concat-wide artifact pass, so they have no ``ArtifactDetectionSource``
    row.

    Raises
    ------
    ValueError
        If zero or both source keys are supplied, if ``sorter`` /
        ``sorter_params_name`` is missing, if a concat source is combined
        with an ``artifact_detection_id``, or if an explicit ``sorting_id``
        does not equal the derived deterministic id.
    """
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
    # Normalize a caller-supplied ``artifact_detection_id`` (which may be a
    # str) so the find-existing comparison is UUID-vs-UUID. A str would
    # otherwise never equal the stored UUID, so an idempotent re-insert
    # would miss its match and create a duplicate sort.
    if artifact_detection_id is not None:
        artifact_detection_id = uuid.UUID(str(artifact_detection_id))

    if has_concat:
        # Concat sorts reuse per-member Recording artifacts; concat-wide
        # artifact detection is a later feature, so a concat request that
        # also asks for an artifact pass is rejected here (the contract is
        # "no ArtifactDetectionSource row" for concat sorts). The identity
        # helper enforces the same invariant; this raise keeps the message
        # at the user-facing insert boundary.
        if artifact_detection_id is not None:
            raise ValueError(
                "SortingSelection.insert_selection: a concat_recording_id "
                "source cannot supply an artifact_detection_id. Concat sorts "
                "reuse per-member Recording artifacts and have no concat-wide "
                "artifact-detection pass."
            )
        source_restriction = {"concat_recording_id": key["concat_recording_id"]}
        identity = sorting_identity_payload(
            concat_recording_id=key["concat_recording_id"],
            sorter=key["sorter"],
            sorter_params_name=key["sorter_params_name"],
        )
        sorting_id = deterministic_id("sorting", identity)
        assert_supplied_id_matches(
            key.get("sorting_id"), sorting_id, field="sorting_id"
        )
        master_row = {**master_restriction, "sorting_id": sorting_id}
        concat_source_row = {"sorting_id": sorting_id, **source_restriction}
        return SortingSelectionPlan(
            sorting_id=sorting_id,
            source_kind="concat",
            master_restriction=master_restriction,
            source_restriction=source_restriction,
            master_row=master_row,
            recording_source_row=None,
            concat_source_row=concat_source_row,
            artifact_source_row=None,
            artifact_detection_id=None,
        )

    source_restriction = {"recording_id": key["recording_id"]}
    # Deterministic, content-addressed sorting_id from the logical identity
    # (recording source + sorter + the optional artifact-detection pass).
    # ``artifact_detection_id=None`` is the single "no artifact-detection
    # pass" form and cannot alias any real artifact_detection_id, so an
    # artifact-detection-backed and an artifact-detection-free sort for the
    # same (recording, sorter) are distinct, idempotent rows.
    identity = sorting_identity_payload(
        recording_id=key["recording_id"],
        sorter=key["sorter"],
        sorter_params_name=key["sorter_params_name"],
        artifact_detection_id=artifact_detection_id,
    )
    sorting_id = deterministic_id("sorting", identity)
    assert_supplied_id_matches(
        key.get("sorting_id"), sorting_id, field="sorting_id"
    )

    master_row = {**master_restriction, "sorting_id": sorting_id}
    recording_source_row = {"sorting_id": sorting_id, **source_restriction}
    artifact_source_row = (
        {
            "sorting_id": sorting_id,
            "artifact_detection_id": artifact_detection_id,
        }
        if artifact_detection_id is not None
        else None
    )
    return SortingSelectionPlan(
        sorting_id=sorting_id,
        source_kind="recording",
        master_restriction=master_restriction,
        source_restriction=source_restriction,
        master_row=master_row,
        recording_source_row=recording_source_row,
        concat_source_row=None,
        artifact_source_row=artifact_source_row,
        artifact_detection_id=artifact_detection_id,
    )


def build_artifact_detection_selection_plan(
    key: dict,
) -> ArtifactDetectionSelectionPlan:
    """Validate an ``ArtifactDetectionSelection`` request and build its plan.

    Reads exactly one source key (``recording_id`` xor
    ``shared_artifact_group_name``); ``artifact_detection_params_name`` is
    required. ``source_kind`` is explicit so a recording source and a
    shared-group source never alias even if their source-identifier strings
    collide.

    Raises
    ------
    ValueError
        If zero or both source keys are supplied, if
        ``artifact_detection_params_name`` is missing, or if an explicit
        ``artifact_detection_id`` does not equal the derived deterministic
        id.
    """
    has_recording = "recording_id" in key
    has_shared = "shared_artifact_group_name" in key
    if has_recording == has_shared:
        raise ValueError(
            "ArtifactDetectionSelection.insert_selection requires exactly one "
            "source key. Provide either recording_id (single-recording "
            "path) or shared_artifact_group_name (cross-recording "
            "path), not both and not neither. Got: "
            f"recording_id={'set' if has_recording else 'unset'}, "
            f"shared_artifact_group_name="
            f"{'set' if has_shared else 'unset'}."
        )

    master_field = "artifact_detection_params_name"
    if master_field not in key:
        raise ValueError(
            "ArtifactDetectionSelection.insert_selection requires "
            f"{master_field!r} in key."
        )
    master_restriction = {master_field: key[master_field]}

    if has_recording:
        source_kind = "recording"
        source_restriction = {"recording_id": key["recording_id"]}
    else:
        source_kind = "shared_artifact_group"
        source_restriction = {
            "shared_artifact_group_name": key["shared_artifact_group_name"]
        }

    # Deterministic, content-addressed artifact_detection_id from the logical
    # identity (params + source), via the shared payload helper so preflight
    # derives the identical id.
    identity = artifact_detection_identity_payload(
        artifact_detection_params_name=key[master_field],
        recording_id=key.get("recording_id"),
        shared_artifact_group_name=key.get("shared_artifact_group_name"),
    )
    artifact_detection_id = deterministic_id("artifact_detection", identity)
    assert_supplied_id_matches(
        key.get("artifact_detection_id"),
        artifact_detection_id,
        field="artifact_detection_id",
    )

    master_row = {
        **master_restriction,
        "artifact_detection_id": artifact_detection_id,
    }
    source_row = {
        "artifact_detection_id": artifact_detection_id,
        **source_restriction,
    }
    return ArtifactDetectionSelectionPlan(
        artifact_detection_id=artifact_detection_id,
        source_kind=source_kind,
        master_restriction=master_restriction,
        source_restriction=source_restriction,
        master_row=master_row,
        source_row=source_row,
    )
