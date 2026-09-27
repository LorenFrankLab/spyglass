"""Unit tests for the pure selection-plan builders.

``SortingSelection.insert_selection`` delegates its pure half to
``_selection_plan``: validate the request, normalize ids, derive the
deterministic content-addressed PK, and shape the master + source part rows.
These tests drive those builders directly -- no
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
    assert plan.artifact_detection_id is None


def test_sorting_plan_with_artifact_normalizes_str_to_uuid():
    """A str ``artifact_detection_id`` is normalized to UUID and threaded into
    the plan; the artifact-backed id differs from the artifact-free id for the
    same (recording, sorter). (The ``ArtifactDetectionSource`` part row itself
    is built DB-side in ``insert_selection`` from the resolved merge id, not in
    the pure plan.)"""
    base = {"recording_id": _REC, "sorter": "ms5", "sorter_params_name": "d"}
    plan = build_sorting_selection_plan({**base, "artifact_detection_id": _ART})
    assert plan.artifact_detection_id == uuid.UUID(_ART)
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


@pytest.mark.parametrize(
    "source", [{"recording_id": _REC}, {"concat_recording_id": _CONCAT}]
)
def test_sorting_plan_rejects_unknown_fields(source):
    """A misspelled optional key (here the corrected recording's) is refused
    by name instead of being dropped, which would select an uncorrected
    sort; the message lists every accepted field."""
    with pytest.raises(ValueError, match="unknown field") as excinfo:
        build_sorting_selection_plan(
            {
                **source,
                "sorter": "ms5",
                "sorter_params_name": "d",
                "motion_corrected_recording": _ART,
                "nwb_file_name": "x.nwb",
            }
        )
    message = str(excinfo.value)
    assert "['motion_corrected_recording', 'nwb_file_name']" in message
    for field in (
        "artifact_detection_id",
        "concat_recording_id",
        "motion_corrected_recording_id",
        "recording_id",
        "sorter",
        "sorter_params_name",
        "sorting_id",
    ):
        assert repr(field) in message


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


# ---------- motion-corrected sort identity ---------------------------------

# Sorting ids and canonical identities recorded before corrected sorts
# existed, for the smoke session's shank-0 recording (MS5 franklab row). An
# uncorrected sort (no motion correction, or only a saved estimate) must keep
# exactly these.
_BASE_RECORDING_ID = "7ec72a85-b8cd-5a57-8977-710fc5b429fe"
_BASE_ARTIFACT_ID = "33c83cb3-6338-50d3-81b7-41e6dc14fc92"
_BASE_CONCAT_ID = "84ff3d7f-4bca-5755-8262-f7bc11c7921f"
_BASE_SORTER = {
    "sorter": "mountainsort5",
    "sorter_params_name": "franklab_30khz_ms5_2026_06",
}
_BASE_SORTS = [
    (
        {
            "recording_id": _BASE_RECORDING_ID,
            "artifact_detection_id": _BASE_ARTIFACT_ID,
        },
        "03a2de42-5552-5fc3-a7a5-bda364732f57",
        '{"artifact_detection_id":"33c83cb3-6338-50d3-81b7-41e6dc14fc92",'
        '"recording_id":"7ec72a85-b8cd-5a57-8977-710fc5b429fe",'
        '"sorter":"mountainsort5",'
        '"sorter_params_name":"franklab_30khz_ms5_2026_06",'
        '"source_kind":"recording"}',
    ),
    (
        {"recording_id": _BASE_RECORDING_ID},
        "f90892e2-4ed3-5c7d-b671-5e3df848778e",
        '{"artifact_detection_id":null,'
        '"recording_id":"7ec72a85-b8cd-5a57-8977-710fc5b429fe",'
        '"sorter":"mountainsort5",'
        '"sorter_params_name":"franklab_30khz_ms5_2026_06",'
        '"source_kind":"recording"}',
    ),
    (
        {"concat_recording_id": _BASE_CONCAT_ID},
        "09f9bdd4-f90c-5c8f-b64d-18c3ea2a66a8",
        '{"concat_recording_id":"84ff3d7f-4bca-5755-8262-f7bc11c7921f",'
        '"sorter":"mountainsort5",'
        '"sorter_params_name":"franklab_30khz_ms5_2026_06",'
        '"source_kind":"concat"}',
    ),
]
_CORRECTED = "44444444-4444-4444-4444-444444444444"
_CORRECTED_2 = "55555555-5555-5555-5555-555555555555"


@pytest.mark.parametrize(("source", "sorting_id", "canonical"), _BASE_SORTS)
def test_off_and_estimate_preserve_sort_input(source, sorting_id, canonical):
    """Without a corrected recording the payload has no motion term and the
    ids equal the recorded ones; an explicit ``None`` is the same request."""
    from spyglass.spikesorting.v2._selection_identity import (
        canonical_identity,
        sorting_identity_payload,
    )

    payload = sorting_identity_payload(**_BASE_SORTER, **source)
    assert "motion_corrected_recording_id" not in payload
    assert canonical_identity(payload) == canonical
    assert (
        canonical_identity(
            sorting_identity_payload(
                **_BASE_SORTER, **source, motion_corrected_recording_id=None
            )
        )
        == canonical
    )
    for key in (
        {**_BASE_SORTER, **source},
        {**_BASE_SORTER, **source, "motion_corrected_recording_id": None},
    ):
        plan = build_sorting_selection_plan(key)
        assert plan.sorting_id == uuid.UUID(sorting_id)
        assert plan.motion_corrected_recording_id is None


@pytest.mark.parametrize(("source", "sorting_id", "canonical"), _BASE_SORTS)
def test_corrected_recording_enters_sort_identity(
    source, sorting_id, canonical
):
    """A corrected recording adds one normalized term to either source kind:
    a new id, distinct per corrected recording, shared by str and UUID."""
    from spyglass.spikesorting.v2._selection_identity import (
        sorting_identity_payload,
    )

    key = {**_BASE_SORTER, **source}
    corrected = build_sorting_selection_plan(
        {**key, "motion_corrected_recording_id": _CORRECTED}
    )
    assert corrected.motion_corrected_recording_id == uuid.UUID(_CORRECTED)
    assert corrected.sorting_id != uuid.UUID(sorting_id)
    assert (
        corrected.source_kind == build_sorting_selection_plan(key).source_kind
    )
    assert (
        build_sorting_selection_plan(
            {**key, "motion_corrected_recording_id": uuid.UUID(_CORRECTED)}
        ).sorting_id
        == corrected.sorting_id
    )
    assert (
        build_sorting_selection_plan(
            {**key, "motion_corrected_recording_id": _CORRECTED_2}
        ).sorting_id
        != corrected.sorting_id
    )
    payload = sorting_identity_payload(
        **key, motion_corrected_recording_id=_CORRECTED
    )
    assert payload == {
        **sorting_identity_payload(**key),
        "motion_corrected_recording_id": uuid.UUID(_CORRECTED),
    }


def test_corrected_concat_sort_still_rejects_an_artifact():
    with pytest.raises(ValueError, match="artifact"):
        build_sorting_selection_plan(
            {
                **_BASE_SORTER,
                "concat_recording_id": _BASE_CONCAT_ID,
                "artifact_detection_id": _BASE_ARTIFACT_ID,
                "motion_corrected_recording_id": _CORRECTED,
            }
        )


def _lineage_of(source):
    from spyglass.spikesorting.v2._source_resolution import SourceLineage

    if "concat_recording_id" in source:
        return SourceLineage(
            kind="concatenated_recording",
            key={
                "concat_recording_id": uuid.UUID(source["concat_recording_id"])
            },
            artifact_detection_id=None,
        )
    detection = source.get("artifact_detection_id")
    return SourceLineage(
        kind="recording",
        key={"recording_id": uuid.UUID(source["recording_id"])},
        artifact_detection_id=(
            None if detection is None else uuid.UUID(detection)
        ),
    )


@pytest.mark.parametrize(("source", "sorting_id", "canonical"), _BASE_SORTS)
def test_sort_parts_must_still_give_the_stored_sorting_id(
    source, sorting_id, canonical
):
    """The parts a selection was inserted with give back its ``sorting_id``
    (the recorded ids included); a correction part added to an uncorrected
    sort, or deleted from a corrected one, gives another id."""
    from spyglass.spikesorting.v2._source_resolution import (
        sorting_parts_mismatch,
    )

    lineage = _lineage_of(source)
    assert (
        sorting_parts_mismatch(
            sorting_id,
            lineage,
            **_BASE_SORTER,
            motion_corrected_recording_id=None,
        )
        is None
    )
    added = sorting_parts_mismatch(
        uuid.UUID(sorting_id),
        lineage,
        **_BASE_SORTER,
        motion_corrected_recording_id=uuid.UUID(_CORRECTED),
    )
    assert added is not None and _CORRECTED in added

    corrected_id = build_sorting_selection_plan(
        {**_BASE_SORTER, **source, "motion_corrected_recording_id": _CORRECTED}
    ).sorting_id
    assert (
        sorting_parts_mismatch(
            corrected_id,
            lineage,
            **_BASE_SORTER,
            motion_corrected_recording_id=_CORRECTED,
        )
        is None
    )
    assert (
        sorting_parts_mismatch(
            corrected_id,
            lineage,
            **_BASE_SORTER,
            motion_corrected_recording_id=None,
        )
        is not None
    )
    assert (
        sorting_parts_mismatch(
            corrected_id,
            lineage,
            **_BASE_SORTER,
            motion_corrected_recording_id=_CORRECTED_2,
        )
        is not None
    )


def test_sort_parts_detect_artifact_part_drift():
    """Deleting a sort's artifact detection part, adding one, or adding one
    to a concatenated-recording sort all break the stored ``sorting_id``."""
    from spyglass.spikesorting.v2._source_resolution import (
        sorting_parts_mismatch,
    )

    (
        (masked, masked_id, _),
        (unmasked, unmasked_id, _),
        (concat, concat_id, _),
    ) = _BASE_SORTS
    masked_lineage = _lineage_of(masked)
    for stored, lineage in (
        (masked_id, masked_lineage._replace(artifact_detection_id=None)),
        (unmasked_id, masked_lineage),
        (
            concat_id,
            _lineage_of(concat)._replace(
                artifact_detection_id=uuid.UUID(_BASE_ARTIFACT_ID)
            ),
        ),
    ):
        assert (
            sorting_parts_mismatch(
                stored,
                lineage,
                **_BASE_SORTER,
                motion_corrected_recording_id=None,
            )
            is not None
        )
    other_sorter = {**_BASE_SORTER, "sorter_params_name": "other"}
    assert (
        sorting_parts_mismatch(
            unmasked_id,
            _lineage_of(unmasked),
            **other_sorter,
            motion_corrected_recording_id=None,
        )
        is not None
    )
