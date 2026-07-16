"""Artifact-detection ``IntervalList`` naming convention (DB-free).

The artifact-removed ``valid_times`` row an ``ArtifactDetection`` run writes to
``IntervalList`` is named ``f"artifact_detection_{artifact_detection_id}"``.
:func:`artifact_detection_interval_list_name` builds that name and
:func:`parse_artifact_detection_interval_list_name` is its inverse; the
``artifact_detection_`` prefix lives here as the single source of truth.

DEPENDENCY-LIGHT BY CONTRACT. This module opens no database connection and
activates no ``dj.schema`` at import: it is pure string handling over the
naming convention, so the source-routing service module (``_curation_routing``)
and a DB-isolated worker can import the parser without a database. ``utils``
re-exports all three names so existing
``from ...v2.utils import artifact_detection_interval_list_name`` (etc.) call
sites are unchanged.
"""

from __future__ import annotations

_ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX = "artifact_detection_"


def artifact_detection_interval_list_name(artifact_detection_id) -> str:
    """Return the ``IntervalList.interval_list_name`` for an artifact detection.

    Centralizes the convention
    ``f"artifact_detection_{artifact_detection_id}"`` so the prefix lives in
    one place; ``parse_artifact_detection_interval_list_name`` is its inverse.

    The ``artifact_detection_`` prefix is intentional -- it disambiguates
    artifact-detection-derived IntervalList rows from sort_valid_times / lfp /
    etc. rows when grepping by name. A query that looks rows up by the bare
    UUID returns empty; use this helper (or its inverse) instead.
    """
    return f"{_ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX}{artifact_detection_id}"


def parse_artifact_detection_interval_list_name(name: str):
    """Return the artifact-detection id encoded in an IntervalList name.

    Returns ``None`` if ``name`` is not in the artifact-named form,
    matching the merge-dispatcher's "leave non-artifact names alone"
    contract.
    """
    if isinstance(name, str) and name.startswith(
        _ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX
    ):
        return name[len(_ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX) :]
    return None
