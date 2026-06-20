"""Unit tests for the pure selection-plan builders.

``SortingSelection`` / ``ArtifactDetectionSelection`` ``insert_selection``
delegate their pure half to ``_selection_plan``: validate the request,
normalize ids, derive the deterministic content-addressed PK, and shape the
master + source part rows. These tests drive those builders directly -- no
DataJoint, no transaction -- so the user-misconfiguration paths (bad source
combinations, missing required keys, supplied-id mismatch, str-vs-UUID
normalization) are pinned cheaply.

The module under test imports only ``_selection_identity`` (DB-free), so
this file needs no database fixture; the import-boundary contract is
separately enforced by ``test_service_modules`` (``_selection_plan`` is in
``_DB_FREE_SERVICE_MODULES``).
"""

from __future__ import annotations

import uuid

import pytest

from spyglass.spikesorting.v2._selection_plan import (
    build_artifact_detection_selection_plan,
    build_sorting_selection_plan,
)

_REC = "11111111-1111-1111-1111-111111111111"
_ART = "22222222-2222-2222-2222-222222222222"


# ---------- build_sorting_selection_plan -----------------------------------


def test_sorting_plan_recording_only_shapes_rows():
    """A recording-only request yields master + recording source rows and no
    artifact source row."""
    plan = build_sorting_selection_plan(
        {"recording_id": _REC, "sorter": "ms5", "sorter_params_name": "default"}
    )
    assert isinstance(plan.sorting_id, uuid.UUID)
    assert plan.master_restriction == {
        "sorter": "ms5",
        "sorter_params_name": "default",
    }
    assert plan.source_restriction == {"recording_id": _REC}
    assert plan.master_row == {
        "sorter": "ms5",
        "sorter_params_name": "default",
        "sorting_id": plan.sorting_id,
    }
    assert plan.recording_source_row == {
        "sorting_id": plan.sorting_id,
        "recording_id": _REC,
    }
    assert plan.artifact_source_row is None
    assert plan.artifact_detection_id is None


def test_sorting_plan_with_artifact_normalizes_str_to_uuid():
    """A str ``artifact_detection_id`` is normalized to UUID and threaded into
    the artifact source row; the artifact-backed id differs from the
    artifact-free id for the same (recording, sorter)."""
    base = {"recording_id": _REC, "sorter": "ms5", "sorter_params_name": "d"}
    plan = build_sorting_selection_plan(
        {**base, "artifact_detection_id": _ART}
    )
    assert plan.artifact_detection_id == uuid.UUID(_ART)
    assert plan.artifact_source_row == {
        "sorting_id": plan.sorting_id,
        "artifact_detection_id": uuid.UUID(_ART),
    }
    # The "no artifact pass" form is a distinct, non-aliasing identity.
    no_art = build_sorting_selection_plan(base)
    assert no_art.sorting_id != plan.sorting_id


def test_sorting_plan_str_and_uuid_artifact_id_share_identity():
    """A str and a UUID ``artifact_detection_id`` produce the same sorting_id
    (the normalization keeps idempotent re-inserts from forking)."""
    base = {"recording_id": _REC, "sorter": "ms5", "sorter_params_name": "d"}
    as_str = build_sorting_selection_plan(
        {**base, "artifact_detection_id": _ART}
    )
    as_uuid = build_sorting_selection_plan(
        {**base, "artifact_detection_id": uuid.UUID(_ART)}
    )
    assert as_str.sorting_id == as_uuid.sorting_id


def test_sorting_plan_is_deterministic():
    """Identical requests derive the identical sorting_id."""
    key = {"recording_id": _REC, "sorter": "ms5", "sorter_params_name": "d"}
    assert (
        build_sorting_selection_plan(key).sorting_id
        == build_sorting_selection_plan(dict(key)).sorting_id
    )


def test_sorting_plan_requires_exactly_one_source():
    """Zero or both source keys raise ValueError."""
    with pytest.raises(ValueError, match="exactly one"):
        build_sorting_selection_plan(
            {"sorter": "ms5", "sorter_params_name": "d"}
        )
    with pytest.raises(ValueError, match="exactly one"):
        build_sorting_selection_plan(
            {
                "recording_id": _REC,
                "concat_recording_id": _REC,
                "sorter": "ms5",
                "sorter_params_name": "d",
            }
        )


def test_sorting_plan_concat_not_implemented():
    """A concat source is rejected with NotImplementedError."""
    with pytest.raises(NotImplementedError, match="concat"):
        build_sorting_selection_plan(
            {
                "concat_recording_id": _REC,
                "sorter": "ms5",
                "sorter_params_name": "d",
            }
        )


def test_sorting_plan_requires_sorter_and_params_name():
    """Missing ``sorter`` or ``sorter_params_name`` raises ValueError."""
    with pytest.raises(ValueError, match="'sorter'"):
        build_sorting_selection_plan(
            {"recording_id": _REC, "sorter_params_name": "d"}
        )
    with pytest.raises(ValueError, match="'sorter_params_name'"):
        build_sorting_selection_plan({"recording_id": _REC, "sorter": "ms5"})


def test_sorting_plan_supplied_id_mismatch_raises_but_match_ok():
    """A supplied sorting_id must equal the derived id; a match is accepted."""
    key = {"recording_id": _REC, "sorter": "ms5", "sorter_params_name": "d"}
    derived = build_sorting_selection_plan(key).sorting_id
    # The correct id is accepted (no raise) and round-trips.
    again = build_sorting_selection_plan(
        {**key, "sorting_id": str(derived)}
    )
    assert again.sorting_id == derived
    # A wrong id is rejected.
    with pytest.raises(ValueError, match="sorting_id"):
        build_sorting_selection_plan({**key, "sorting_id": _ART})


# ---------- build_artifact_detection_selection_plan ------------------------


def test_artifact_plan_recording_source_shapes_rows():
    """A recording-source request yields source_kind='recording' + rows."""
    plan = build_artifact_detection_selection_plan(
        {"recording_id": _REC, "artifact_detection_params_name": "default"}
    )
    assert plan.source_kind == "recording"
    assert isinstance(plan.artifact_detection_id, uuid.UUID)
    assert plan.master_restriction == {
        "artifact_detection_params_name": "default"
    }
    assert plan.source_restriction == {"recording_id": _REC}
    assert plan.master_row == {
        "artifact_detection_params_name": "default",
        "artifact_detection_id": plan.artifact_detection_id,
    }
    assert plan.source_row == {
        "artifact_detection_id": plan.artifact_detection_id,
        "recording_id": _REC,
    }


def test_artifact_plan_shared_group_source_shapes_rows():
    """A shared-group request yields source_kind='shared_artifact_group'."""
    plan = build_artifact_detection_selection_plan(
        {
            "shared_artifact_group_name": "grp1",
            "artifact_detection_params_name": "default",
        }
    )
    assert plan.source_kind == "shared_artifact_group"
    assert plan.source_restriction == {"shared_artifact_group_name": "grp1"}
    assert plan.source_row == {
        "artifact_detection_id": plan.artifact_detection_id,
        "shared_artifact_group_name": "grp1",
    }


def test_artifact_plan_recording_and_shared_group_dont_alias():
    """The two source kinds never collide even on identical id strings."""
    rec = build_artifact_detection_selection_plan(
        {"recording_id": "x", "artifact_detection_params_name": "p"}
    )
    grp = build_artifact_detection_selection_plan(
        {"shared_artifact_group_name": "x", "artifact_detection_params_name": "p"}
    )
    assert rec.artifact_detection_id != grp.artifact_detection_id


def test_artifact_plan_requires_exactly_one_source():
    """Zero or both source keys raise ValueError."""
    with pytest.raises(ValueError, match="exactly one"):
        build_artifact_detection_selection_plan(
            {"artifact_detection_params_name": "default"}
        )
    with pytest.raises(ValueError, match="exactly one"):
        build_artifact_detection_selection_plan(
            {
                "recording_id": _REC,
                "shared_artifact_group_name": "g",
                "artifact_detection_params_name": "default",
            }
        )


def test_artifact_plan_requires_params_name():
    """Missing ``artifact_detection_params_name`` raises ValueError."""
    with pytest.raises(ValueError, match="artifact_detection_params_name"):
        build_artifact_detection_selection_plan({"recording_id": _REC})


def test_artifact_plan_supplied_id_mismatch_raises_but_match_ok():
    """A supplied artifact_detection_id must equal the derived id."""
    key = {"recording_id": _REC, "artifact_detection_params_name": "default"}
    derived = build_artifact_detection_selection_plan(key).artifact_detection_id
    again = build_artifact_detection_selection_plan(
        {**key, "artifact_detection_id": str(derived)}
    )
    assert again.artifact_detection_id == derived
    with pytest.raises(ValueError, match="artifact_detection_id"):
        build_artifact_detection_selection_plan(
            {**key, "artifact_detection_id": _ART}
        )
