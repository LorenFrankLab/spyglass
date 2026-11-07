"""Pure source-routing classifier behind ``CurationV2.resolve_restriction``.

Splits a restriction into per-source dicts; the join assembly is
:func:`._curation.restriction.resolve_restriction`.
"""

from __future__ import annotations

import uuid
from typing import NamedTuple

from spyglass.spikesorting.v2._artifacts.naming import (
    parse_artifact_detection_interval_list_name,
)
from spyglass.utils import logger

# Only on ``RecordingSelection`` (via ``SortingSelection.RecordingSource``).
_REC_KEYS = (
    "nwb_file_name",
    "team_name",
    "sort_group_id",
    "interval_list_name",
    "recording_id",
)
# Via ``ConcatenatedRecordingSelection`` -> ``SessionGroup``.
_CONCAT_KEYS = (
    "concat_recording_id",
    "session_group_owner",
    "session_group_name",
)
# On both source selections; alone, matches both families.
_SHARED_SOURCE_KEYS = ("preprocessing_params_name",)
# Artifact and motion ids live on optional parts, so the join assembly
# resolves them separately (``_PART_RESOLVED_SORT_KEYS``).
_SORT_KEYS = (
    "sorter",
    "sorter_params_name",
    "sorting_id",
    "artifact_detection_id",
    "motion_corrected_recording_id",
)
_PART_RESOLVED_SORT_KEYS = (
    "artifact_detection_id",
    "motion_corrected_recording_id",
)
_CURATION_KEYS = ("curation_id",)
_ALLOWED_KEYS = frozenset(
    _REC_KEYS + _CONCAT_KEYS + _SHARED_SOURCE_KEYS + _SORT_KEYS + _CURATION_KEYS
)

# Absent-key wildcards, distinct from an explicit ``None``.
NO_ARTIFACT_RESTRICTION = object()
NO_MOTION_CORRECTION_RESTRICTION = object()


class RestrictionPlan(NamedTuple):
    """The normalized pieces ``resolve_restriction``'s join assembly consumes.

    - ``*_restriction``: keys routing to that table; may be ``{}``.
      ``shared_restriction`` applies to whichever source family routes.
    - ``artifact_detection_id``: a ``uuid.UUID`` matches standalone or
      concat-member detections, ``None`` excludes both, and
      :data:`NO_ARTIFACT_RESTRICTION` is a wildcard.
    - ``motion_corrected_recording_id``: a ``uuid.UUID`` matches that
      corrected recording, ``None`` uncorrected sorts, and
      :data:`NO_MOTION_CORRECTION_RESTRICTION` both.
    - ``unresolved_name_warning``: emitted by the join assembly when
      ``restrict_by_artifact`` found no artifact id, else ``None``.
    """

    rec_restriction: dict
    concat_restriction: dict
    shared_restriction: dict
    sort_restriction: dict
    curation_restriction: dict
    artifact_detection_id: object
    motion_corrected_recording_id: object
    unresolved_name_warning: "str | None"


def classify_and_normalize_restriction(
    key: dict,
    *,
    restrict_by_artifact: bool,
    strict: bool,
) -> "RestrictionPlan | None":
    """Classify+normalize a user restriction into the per-source dicts.

    Returns ``None`` for an unknown key when ``strict=False``. Raises
    ``ValueError`` for an unknown key when ``strict=True`` or for mixed
    concat and single-recording keys.
    """
    unknown = set(key) - _ALLOWED_KEYS
    if unknown:
        if not strict:
            # Another pipeline's key: contribute no v2 rows.
            return None
        raise ValueError(
            "CurationV2.resolve_restriction: "
            f"unknown restriction keys {sorted(unknown)}. Allowed: "
            f"{sorted(_ALLOWED_KEYS)}."
        )

    key = dict(key)
    unresolved_name_warning = None
    # Map ``artifact_detection_{uuid}`` interval names to the id column.
    if restrict_by_artifact and "interval_list_name" in key:
        artifact_detection_id = parse_artifact_detection_interval_list_name(
            key["interval_list_name"]
        )
        if artifact_detection_id is not None:
            key["artifact_detection_id"] = artifact_detection_id
            key.pop("interval_list_name", None)
        elif "artifact_detection_id" not in key:
            # Warn rather than silently return unrestricted ids.
            unresolved_name_warning = (
                "CurationV2.resolve_restriction: "
                "restrict_by_artifact=True but interval_list_name "
                f"{key['interval_list_name']!r} is not an "
                "'artifact_detection_{uuid}' interval and no "
                "artifact_detection_id was given; v2 results will NOT be "
                "artifact-restricted. Pass artifact_detection_id=... or "
                "use the artifact-detection interval to restrict."
            )

    # A malformed id fails here rather than matching nothing.
    if key.get("artifact_detection_id") is not None:
        key["artifact_detection_id"] = uuid.UUID(
            str(key["artifact_detection_id"])
        )
    if key.get("motion_corrected_recording_id") is not None:
        key["motion_corrected_recording_id"] = uuid.UUID(
            str(key["motion_corrected_recording_id"])
        )

    concat_restriction = {k: key[k] for k in _CONCAT_KEYS if k in key}
    rec_restriction = {k: key[k] for k in _REC_KEYS if k in key}
    shared_restriction = {k: key[k] for k in _SHARED_SOURCE_KEYS if k in key}
    if concat_restriction and rec_restriction:
        # The caller never sees the warning on this raising path.
        if unresolved_name_warning is not None:
            logger.warning(unresolved_name_warning)
        raise ValueError(
            "CurationV2.resolve_restriction: cannot combine concat-source "
            f"keys {sorted(concat_restriction)} with single-recording "
            f"keys {sorted(rec_restriction)}; a sort has exactly one input "
            "source. Restrict by one source family."
        )

    sort_restriction = {
        k: key[k]
        for k in _SORT_KEYS
        if k in key and k not in _PART_RESOLVED_SORT_KEYS
    }
    curation_restriction = {k: key[k] for k in _CURATION_KEYS if k in key}

    if "artifact_detection_id" in key:
        artifact_detection_id = key["artifact_detection_id"]
    else:
        artifact_detection_id = NO_ARTIFACT_RESTRICTION
    motion_corrected_recording_id = key.get(
        "motion_corrected_recording_id", NO_MOTION_CORRECTION_RESTRICTION
    )

    return RestrictionPlan(
        rec_restriction=rec_restriction,
        concat_restriction=concat_restriction,
        shared_restriction=shared_restriction,
        sort_restriction=sort_restriction,
        curation_restriction=curation_restriction,
        artifact_detection_id=artifact_detection_id,
        motion_corrected_recording_id=motion_corrected_recording_id,
        unresolved_name_warning=unresolved_name_warning,
    )
