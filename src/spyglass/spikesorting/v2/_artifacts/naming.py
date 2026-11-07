"""Artifact-detection ``IntervalList`` naming convention.

Names are ``f"artifact_detection_{artifact_detection_id}"``.
"""

from __future__ import annotations

_ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX = "artifact_detection_"


def artifact_detection_interval_list_name(artifact_detection_id) -> str:
    """Return the ``IntervalList.interval_list_name`` for an artifact detection.

    Inverse of ``parse_artifact_detection_interval_list_name``. Lookups by the
    bare UUID return nothing.
    """
    return f"{_ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX}{artifact_detection_id}"


def parse_artifact_detection_interval_list_name(name: str):
    """Return the artifact-detection id encoded in an IntervalList name.

    Returns ``None`` if ``name`` is not in the artifact-named form.
    """
    if isinstance(name, str) and name.startswith(
        _ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX
    ):
        return name[len(_ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX) :]
    return None
