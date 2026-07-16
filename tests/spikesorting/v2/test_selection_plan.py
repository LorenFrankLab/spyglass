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
separately enforced by ``test_service_import_contracts`` (``_selection_plan`` is
in ``_DB_FREE_SERVICE_MODULES``).
"""

from __future__ import annotations

import uuid

import pytest

from spyglass.spikesorting.v2._selection_plan import (
    build_artifact_detection_selection_plan,
    build_recording_selection_plan,
    build_sorting_selection_plan,
)

# Every test in this module drives the pure ``_selection_plan`` builders
# directly -- no DataJoint connection, no I/O -- so the whole file qualifies as
# ``unit`` and is collected (not skipped) under ``pytest -m unit``.
pytestmark = pytest.mark.unit

_REC = "11111111-1111-1111-1111-111111111111"
_ART = "22222222-2222-2222-2222-222222222222"
_CONCAT = "33333333-3333-3333-3333-333333333333"

_FULL_REC = {
    "nwb_file_name": "mearec.nwb",
    "sort_group_id": 0,
    "interval_list_name": "raw data valid times",
    "preprocessing_params_name": "default",
    "team_name": "test_team",
}
# A resolved recording_input_hash (the DB-side resolution is exercised in the
# integration tests); here any fixed 64-hex value drives the pure shaping.
_REC_HASH = "a" * 64
_REC_HASH_2 = "b" * 64


# ---------- build_recording_selection_plan ---------------------------------


def test_recording_plan_shapes_master_row_and_restriction():
    """A full FK set + resolved input hash yields the identity restriction
    (FK set + hash) plus a master row carrying the deterministic recording_id;
    the FK set + hash IS the find-existing restriction."""
    plan = build_recording_selection_plan(
        dict(_FULL_REC), recording_input_hash=_REC_HASH
    )
    assert isinstance(plan.recording_id, uuid.UUID)
    assert plan.master_restriction == {
        **_FULL_REC,
        "recording_input_hash": _REC_HASH,
    }
    assert plan.master_row == {
        **_FULL_REC,
        "recording_input_hash": _REC_HASH,
        "recording_id": plan.recording_id,
    }


def test_recording_plan_is_deterministic():
    """Identical FK sets + hash derive the identical recording_id."""
    assert (
        build_recording_selection_plan(
            dict(_FULL_REC), recording_input_hash=_REC_HASH
        ).recording_id
        == build_recording_selection_plan(
            dict(_FULL_REC), recording_input_hash=_REC_HASH
        ).recording_id
    )


def test_recording_plan_input_hash_changes_recording_id():
    """The resolved-input hash is folded into the identity: a changed input
    set (a different hash under the SAME FK set) mints a different
    recording_id. This is the OP-3/OP-4 honesty guarantee."""
    id_a = build_recording_selection_plan(
        dict(_FULL_REC), recording_input_hash=_REC_HASH
    ).recording_id
    id_b = build_recording_selection_plan(
        dict(_FULL_REC), recording_input_hash=_REC_HASH_2
    ).recording_id
    assert id_a != id_b


def test_recording_plan_rejects_unknown_field():
    """An extra (joined/fetched) field is rejected, not silently hashed into
    a different recording_id."""
    with pytest.raises(ValueError, match="unknown field"):
        build_recording_selection_plan(
            {**_FULL_REC, "analysis_file_name": "x"},
            recording_input_hash=_REC_HASH,
        )


def test_recording_plan_requires_all_identity_fields():
    """A missing identity field raises rather than deriving a partial id."""
    incomplete = {k: v for k, v in _FULL_REC.items() if k != "team_name"}
    with pytest.raises(ValueError, match="requires field"):
        build_recording_selection_plan(
            incomplete, recording_input_hash=_REC_HASH
        )


def test_recording_plan_accepts_matching_supplied_id():
    """An explicit recording_id equal to the derived id is accepted."""
    det = build_recording_selection_plan(
        dict(_FULL_REC), recording_input_hash=_REC_HASH
    ).recording_id
    plan = build_recording_selection_plan(
        {**_FULL_REC, "recording_id": det}, recording_input_hash=_REC_HASH
    )
    assert plan.recording_id == det


def test_recording_plan_rejects_mismatched_supplied_id():
    """An explicit recording_id that disagrees with the derived id raises."""
    wrong = uuid.UUID("99999999-9999-9999-9999-999999999999")
    with pytest.raises(ValueError):
        build_recording_selection_plan(
            {**_FULL_REC, "recording_id": wrong},
            recording_input_hash=_REC_HASH,
        )


# ---------- build_sorting_selection_plan -----------------------------------


def test_sorting_plan_recording_only_shapes_rows():
    """A recording-only request yields master + recording source rows and no
    artifact source row."""
    plan = build_sorting_selection_plan(
        {"recording_id": _REC, "sorter": "ms5", "sorter_params_name": "default"}
    )
    assert isinstance(plan.sorting_id, uuid.UUID)
    assert plan.source_kind == "recording"
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
    assert plan.concat_source_row is None
    assert plan.artifact_source_row is None
    assert plan.artifact_detection_id is None


def test_sorting_plan_with_artifact_normalizes_str_to_uuid():
    """A str ``artifact_detection_id`` is normalized to UUID and threaded into
    the artifact source row; the artifact-backed id differs from the
    artifact-free id for the same (recording, sorter)."""
    base = {"recording_id": _REC, "sorter": "ms5", "sorter_params_name": "d"}
    plan = build_sorting_selection_plan({**base, "artifact_detection_id": _ART})
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


def test_sorting_plan_concat_only_shapes_rows():
    """A concat-only request yields master + concat source rows, no recording
    source row, and no artifact source row."""
    plan = build_sorting_selection_plan(
        {
            "concat_recording_id": _CONCAT,
            "sorter": "ms5",
            "sorter_params_name": "default",
        }
    )
    assert isinstance(plan.sorting_id, uuid.UUID)
    assert plan.source_kind == "concat"
    assert plan.master_restriction == {
        "sorter": "ms5",
        "sorter_params_name": "default",
    }
    assert plan.source_restriction == {"concat_recording_id": _CONCAT}
    assert plan.master_row == {
        "sorter": "ms5",
        "sorter_params_name": "default",
        "sorting_id": plan.sorting_id,
    }
    assert plan.concat_source_row == {
        "sorting_id": plan.sorting_id,
        "concat_recording_id": _CONCAT,
    }
    assert plan.recording_source_row is None
    assert plan.artifact_source_row is None
    assert plan.artifact_detection_id is None


def test_sorting_plan_concat_is_deterministic():
    """Identical concat requests derive the identical sorting_id."""
    key = {
        "concat_recording_id": _CONCAT,
        "sorter": "ms5",
        "sorter_params_name": "d",
    }
    assert (
        build_sorting_selection_plan(key).sorting_id
        == build_sorting_selection_plan(dict(key)).sorting_id
    )


def test_sorting_plan_concat_id_change_changes_sorting_id():
    """Changing only the concat_recording_id changes the sorting_id."""
    base = {"sorter": "ms5", "sorter_params_name": "d"}
    a = build_sorting_selection_plan({**base, "concat_recording_id": _CONCAT})
    b = build_sorting_selection_plan({**base, "concat_recording_id": _REC})
    assert a.sorting_id != b.sorting_id


def test_sorting_plan_concat_and_recording_same_uuid_distinct_identity():
    """A recording source and a concat source with the SAME uuid value resolve
    to different sorting_ids -- the input source kind participates in the
    content-addressed identity, so a recording_id can never alias a
    concat_recording_id."""
    base = {"sorter": "ms5", "sorter_params_name": "d"}
    rec = build_sorting_selection_plan({**base, "recording_id": _REC})
    concat = build_sorting_selection_plan({**base, "concat_recording_id": _REC})
    assert rec.sorting_id != concat.sorting_id


def test_sorting_plan_concat_rejects_artifact():
    """A concat request that also supplies an artifact is rejected: concat
    sorts must have no ArtifactDetectionSource row."""
    with pytest.raises(ValueError, match="artifact"):
        build_sorting_selection_plan(
            {
                "concat_recording_id": _CONCAT,
                "sorter": "ms5",
                "sorter_params_name": "d",
                "artifact_detection_id": _ART,
            }
        )


def test_sorting_plan_concat_supplied_id_mismatch_raises_but_match_ok():
    """A supplied sorting_id must equal the derived concat id; a match is
    accepted."""
    key = {
        "concat_recording_id": _CONCAT,
        "sorter": "ms5",
        "sorter_params_name": "d",
    }
    derived = build_sorting_selection_plan(key).sorting_id
    again = build_sorting_selection_plan({**key, "sorting_id": str(derived)})
    assert again.sorting_id == derived
    with pytest.raises(ValueError):
        build_sorting_selection_plan(
            {**key, "sorting_id": "99999999-9999-9999-9999-999999999999"}
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
    again = build_sorting_selection_plan({**key, "sorting_id": str(derived)})
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
        {
            "shared_artifact_group_name": "x",
            "artifact_detection_params_name": "p",
        }
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
