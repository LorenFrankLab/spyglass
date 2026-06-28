"""Pure source-routing classifier behind ``CurationV2.resolve_restriction``.

``CurationV2.resolve_restriction`` interprets a cross-table restriction over
the v2 part-table convention keys and resolves it to the ``CurationV2`` rows it
selects. Its work splits cleanly into two halves: a PURE classification +
normalization half (here) and a DataJoint join-assembly half (the table
method). :func:`classify_and_normalize_restriction` validates the restriction
keys, maps the ``artifact_detection_{uuid}`` ``interval_list_name`` convention
back to ``artifact_detection_id``, normalizes that id to a ``uuid.UUID``, and
splits the remaining keys into the per-source restriction dicts the join
assembly consumes -- without touching the database. The table method emits the
captured unresolved-name warning, then assembles the joins from the returned
dicts exactly as before.

DEPENDENCY-LIGHT BY CONTRACT. This module opens no database connection and
activates no ``dj.schema`` at import: it imports only the standard library and
the DB-free ``_artifact_naming`` parser, so a DB-isolated worker can import it
and run the classification without a database.
"""

from __future__ import annotations

import logging
import uuid
from typing import NamedTuple

from spyglass.spikesorting.v2._artifact_naming import (
    parse_artifact_detection_interval_list_name,
)

logger = logging.getLogger("spyglass")

# Single-recording-ONLY source restriction keys: route through
# ``SortingSelection.RecordingSource`` -> ``RecordingSelection``. These columns
# exist only on the single-recording selection, NOT on
# ``ConcatenatedRecordingSelection`` (see ``_SHARED_SOURCE_KEYS`` for the
# preprocessing recipe, which lives on both).
_REC_KEYS = (
    "nwb_file_name",
    "team_name",
    "sort_group_id",
    "interval_list_name",
    "recording_id",
)
# Concat-source restriction keys: route through
# ``SortingSelection.ConcatenatedRecordingSource`` ->
# ``ConcatenatedRecordingSelection`` -> ``SessionGroup``.
# ``motion_correction_params_name`` is included so a concat restriction can pin
# the motion recipe too.
_CONCAT_KEYS = (
    "concat_recording_id",
    "session_group_owner",
    "session_group_name",
    "motion_correction_params_name",
)
# Cross-source restriction keys: columns present on BOTH ``RecordingSelection``
# and ``ConcatenatedRecordingSelection`` (the preprocessing recipe is shared by
# single-recording and concat sources). A shared key filters whichever source
# family the restriction routes to, and -- with no source-specific key -- must
# match BOTH families (a broad ``{preprocessing_params_name}`` query must not
# silently drop concat-backed curations). It NEVER triggers the
# concat-vs-recording contradiction, since it is not exclusive to either.
_SHARED_SOURCE_KEYS = ("preprocessing_params_name",)
# Sort-level keys (``artifact_detection_id`` lives on the optional
# ``SortingSelection.ArtifactDetectionSource`` part, so it is resolved
# separately from the other sort keys by the join assembly).
_SORT_KEYS = (
    "sorter",
    "sorter_params_name",
    "sorting_id",
    "artifact_detection_id",
)
_CURATION_KEYS = ("curation_id",)
_ALLOWED_KEYS = frozenset(
    _REC_KEYS + _CONCAT_KEYS + _SHARED_SOURCE_KEYS + _SORT_KEYS + _CURATION_KEYS
)

# Sentinel distinguishing "the restriction names no artifact_detection_id"
# (a wildcard -- no artifact restriction) from an explicit
# ``artifact_detection_id=None`` (an anti-join to sorts with no
# ``ArtifactDetectionSource`` row). A bare ``None`` is already taken for the
# anti-join, so absence needs its own marker; only an absent key is a wildcard.
NO_ARTIFACT_RESTRICTION = object()


class RestrictionPlan(NamedTuple):
    """The normalized pieces ``resolve_restriction``'s join assembly consumes.

    Each ``*_restriction`` dict is the subset of the (normalized) restriction
    routing to that source / sort / curation table; any may be ``{}``.
    ``shared_restriction`` carries cross-source keys (the preprocessing recipe)
    present on both source selections: the join assembly applies it to whichever
    family routes, or matches both families when no source-specific key is
    given. ``artifact_detection_id`` is the resolved id the optional
    ``SortingSelection.ArtifactDetectionSource`` part is restricted by: a
    ``uuid.UUID`` intersects that part, ``None`` anti-joins it (sorts with no
    artifact-detection pass), and the :data:`NO_ARTIFACT_RESTRICTION` sentinel
    means the restriction named no artifact id at all (a wildcard -- no
    artifact restriction). ``restrict_by_artifact`` is the caller's flag,
    threaded through unchanged. ``unresolved_name_warning`` is the message the
    table method emits via ``logger.warning`` when ``restrict_by_artifact`` was
    requested but the interval name carried no artifact id (``None`` when there
    is nothing to warn about).
    """

    rec_restriction: dict
    concat_restriction: dict
    shared_restriction: dict
    sort_restriction: dict
    curation_restriction: dict
    artifact_detection_id: object
    restrict_by_artifact: bool
    unresolved_name_warning: "str | None"


def classify_and_normalize_restriction(
    key: dict,
    *,
    restrict_by_artifact: bool,
    strict: bool,
) -> "RestrictionPlan | None":
    """Classify+normalize a user restriction into the per-source dicts.

    Returns the exact per-source restriction dicts the
    ``CurationV2.resolve_restriction`` join assembly consumes, plus the
    ``artifact_detection_id`` (``None`` => the existing anti-join,
    :data:`NO_ARTIFACT_RESTRICTION` => no artifact restriction) and any
    unresolved-name warning. Returns ``None`` on the lenient (``strict=False``)
    unrecognized-key bail-out; raises ``ValueError`` on an unrecognized key
    when ``strict=True`` and on a contradictory concat+recording restriction.
    A cross-source key (the preprocessing recipe) goes into
    ``shared_restriction`` and never triggers that contradiction. Pure: no
    DataJoint, no DB.
    """
    unknown = set(key) - _ALLOWED_KEYS
    if unknown:
        if not strict:
            # Lenient multi-source-dispatch path: an unknown key names a column
            # from another pipeline, so this is not a v2 query. Return ``None``
            # so the caller contributes no v2 rows. The strict raise is
            # reserved for a deliberate v2 query, where an unknown key is a typo.
            return None
        raise ValueError(
            "CurationV2.resolve_restriction: "
            f"unknown restriction keys {sorted(unknown)}. Allowed: "
            f"{sorted(_ALLOWED_KEYS)}."
        )

    key = dict(key)
    unresolved_name_warning = None
    # ``restrict_by_artifact`` maps the artifact-detection IntervalList
    # convention (``f"artifact_detection_{artifact_detection_id}"``) back to the
    # ``artifact_detection_id`` master-side column so the v2 join chain
    # downstream resolves correctly.
    if restrict_by_artifact and "interval_list_name" in key:
        artifact_detection_id = parse_artifact_detection_interval_list_name(
            key["interval_list_name"]
        )
        if artifact_detection_id is not None:
            key["artifact_detection_id"] = artifact_detection_id
            key.pop("interval_list_name", None)
        elif "artifact_detection_id" not in key:
            # The caller asked to restrict by artifact, but the interval name is
            # not the ``artifact_detection_{uuid}`` form and no
            # artifact_detection_id was supplied, so there is nothing to map.
            # Capture a warning (emitted at the table boundary) instead of
            # silently returning unrestricted ids.
            unresolved_name_warning = (
                "CurationV2.resolve_restriction: "
                "restrict_by_artifact=True but interval_list_name "
                f"{key['interval_list_name']!r} is not an "
                "'artifact_detection_{uuid}' interval and no "
                "artifact_detection_id was given; v2 results will NOT be "
                "artifact-restricted. Pass artifact_detection_id=... or "
                "use the artifact-detection interval to restrict."
            )

    # ``parse_artifact_detection_interval_list_name`` returns a str and a caller
    # may pass either a str or a UUID, but ``artifact_detection_id`` is a uuid
    # column. Normalize to a ``uuid.UUID`` so the ArtifactDetectionSource
    # intersection downstream is unambiguous and a malformed id fails fast here
    # rather than silently matching nothing.
    if key.get("artifact_detection_id") is not None:
        key["artifact_detection_id"] = uuid.UUID(
            str(key["artifact_detection_id"])
        )

    # Route through the input source the restriction names. A concat
    # restriction (concat_recording_id / session-group / motion recipe) and a
    # single-recording restriction are mutually exclusive (a sort has exactly
    # one input source), so mixing them is a contradictory restriction and is
    # rejected.
    concat_restriction = {k: key[k] for k in _CONCAT_KEYS if k in key}
    rec_restriction = {k: key[k] for k in _REC_KEYS if k in key}
    # Cross-source keys (the preprocessing recipe) are NOT counted toward the
    # concat-vs-recording contradiction -- they live on both source selections.
    shared_restriction = {k: key[k] for k in _SHARED_SOURCE_KEYS if k in key}
    if concat_restriction and rec_restriction:
        # On the normal (non-raising) path the unresolved-name warning is
        # emitted at the table boundary. This path raises before returning, so
        # the boundary is never reached; emit inline here first to preserve the
        # original warn-then-raise order on this (doubly-degenerate) input.
        if unresolved_name_warning is not None:
            logger.warning(unresolved_name_warning)
        raise ValueError(
            "CurationV2.resolve_restriction: cannot combine concat-source "
            f"keys {sorted(concat_restriction)} with single-recording "
            f"keys {sorted(rec_restriction)}; a sort has exactly one input "
            "source. Restrict by one source family."
        )

    # ``artifact_detection_id`` is resolved through the part by the join
    # assembly, NOT applied directly to ``sort_master``; the other sort keys go
    # on the master.
    sort_restriction = {
        k: key[k]
        for k in _SORT_KEYS
        if k in key and k != "artifact_detection_id"
    }
    curation_restriction = {k: key[k] for k in _CURATION_KEYS if k in key}

    # In the v2 design the presence/absence of an ``ArtifactDetectionSource``
    # row IS the artifact-detection state, so ``artifact_detection_id=None``
    # means "no artifact-detection pass" (anti-join); an absent key is a
    # wildcard (no artifact restriction at all).
    if "artifact_detection_id" in key:
        artifact_detection_id = key["artifact_detection_id"]
    else:
        artifact_detection_id = NO_ARTIFACT_RESTRICTION

    return RestrictionPlan(
        rec_restriction=rec_restriction,
        concat_restriction=concat_restriction,
        shared_restriction=shared_restriction,
        sort_restriction=sort_restriction,
        curation_restriction=curation_restriction,
        artifact_detection_id=artifact_detection_id,
        restrict_by_artifact=restrict_by_artifact,
        unresolved_name_warning=unresolved_name_warning,
    )
