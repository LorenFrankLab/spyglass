"""Deterministic, content-addressed selection identities.

- Pure-helper tests (no DB): canonicalization (``str`` vs ``uuid.UUID``,
  numpy vs int), ``None`` never aliasing a real id, and table ``kind`` /
  ``source_kind`` separating ids.
- DB-tier tests: the id is the written PK, repeated calls are idempotent, a
  mismatched supplied id is rejected, and a duplicate-PK race loser
  refetches the winner's row.
"""

from __future__ import annotations

import uuid
from unittest import mock

import pytest

from spyglass.spikesorting.v2._core.selection_identity import (
    RECORDING_IDENTITY_FIELDS,
    V2_SELECTION_NAMESPACE,
    assert_supplied_id_matches,
    deterministic_id,
    existing_selection_pk,
    recording_identity_payload,
    recording_input_hash,
)
from spyglass.utils.identity import canonical_json, sha256_json

# --------------------------------------------------------------------------
# Pure-helper unit tests (no database).
# --------------------------------------------------------------------------


def test_namespace_is_frozen_literal():
    """The v2 namespace is the frozen uuid5(DNS, seed) literal.

    Changing the seed would change every selection id.
    """
    assert V2_SELECTION_NAMESPACE == uuid.uuid5(
        uuid.NAMESPACE_DNS, "spyglass.spikesorting.v2.selection"
    )
    assert str(V2_SELECTION_NAMESPACE) == "b44d4765-4714-5c69-96d5-97feb2217e86"


def test_stored_ids_outside_evaluation_are_pinned():
    """Sorting, merge, FigPack and UnitMatch ids keep their stored values.

    Values computed on the commit before evaluation ids took the curation's
    ``curation_uuid``.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2._core.selection_identity import (
        sorting_identity_payload,
    )
    from spyglass.spikesorting.v2._matching.graph import input_set_hash
    from spyglass.spikesorting.v2._review.annotations import (
        figpack_config_hash,
    )
    from tests.spikesorting.v2.test_matcher_graph_inputs import _parts

    sorting_id = uuid.UUID(int=11)
    assert (
        str(
            deterministic_id(
                "sorting",
                sorting_identity_payload(
                    recording_id=uuid.UUID(int=2),
                    sorter="mountainsort4",
                    sorter_params_name="franklab_tetrode_hippocampus_30KHz",
                ),
            )
        )
        == "a8e65109-075d-5e04-ac70-f2162adeed74"
    )
    assert (
        dj.hash.key_hash(
            {"sorting_id": sorting_id, "curation_id": 3, "source": "CurationV2"}
        )
        == "3d1f75c7f40ed547fa9e306e70df330f"
    )
    config_hash = figpack_config_hash(
        sorting_id=sorting_id,
        curation_id=3,
        curation_uuid=uuid.UUID(int=12),
        label_options=["accept", "mua", "noise"],
        displayed_unit_properties=None,
        upload=False,
        ephemeral=False,
    )
    assert (
        str(
            deterministic_id(
                "figpack_curation",
                {
                    "sorting_id": sorting_id,
                    "curation_id": 3,
                    "figpack_config_hash": config_hash,
                },
            )
        )
        == "649fa654-4ca0-55eb-8688-8e842cc3162e"
    )
    set_hash = input_set_hash(*_parts())
    assert (
        str(
            deterministic_id(
                "unitmatch",
                {"matcher_params_name": "default", "input_set_hash": set_hash},
            )
        )
        == "2bde63c4-e9c4-58a5-a65a-2a746926b84f"
    )


def test_evaluation_id_follows_the_curation_generation():
    """The same evaluation fields over a replaced curation get a new id."""
    from spyglass.spikesorting.v2._core.selection_identity import (
        evaluation_selection_id,
    )

    identity = {
        "sorting_id": uuid.UUID(int=11),
        "curation_id": 3,
        "metric_params_name": "m",
        "auto_curation_rules_name": "r",
        "metric_waveform_params_name": "w",
        "observation_version": 1,
    }
    first = evaluation_selection_id(identity, uuid.UUID(int=12))
    assert first == evaluation_selection_id(identity, str(uuid.UUID(int=12)))
    assert first != evaluation_selection_id(identity, uuid.UUID(int=13))
    # The id before the curation_uuid joined the identity.
    assert first != uuid.UUID("b92e5f0a-9928-582b-aadf-a932593b9341")
    assert first != deterministic_id("curation_evaluation", identity)


def test_deterministic_id_returns_uuid():
    assert isinstance(deterministic_id("recording", {"a": 1}), uuid.UUID)


def test_deterministic_id_stable_across_uuid_vs_str():
    """A ``str`` and a ``uuid.UUID`` of the same value hash identically.

    Otherwise a str ``artifact_detection_id`` would fork a duplicate sort
    (the DB-level case is
    ``test_insert_selection_dedup_accepts_str_artifact_detection_id``).
    """
    u = uuid.uuid4()
    assert deterministic_id(
        "sorting", {"artifact_detection_id": u}
    ) == deterministic_id("sorting", {"artifact_detection_id": str(u)})


def test_deterministic_id_stable_across_uuid_case():
    """Upper- and lower-case UUID strings canonicalize to one id."""
    u = uuid.uuid4()
    assert deterministic_id(
        "sorting", {"artifact_detection_id": str(u).upper()}
    ) == deterministic_id("sorting", {"artifact_detection_id": str(u)})


def test_deterministic_id_independent_of_key_order():
    assert deterministic_id(
        "recording", {"a": 1, "b": "x"}
    ) == deterministic_id("recording", {"b": "x", "a": 1})


def test_deterministic_id_repeated_call_stable():
    payload = {"nwb_file_name": "s_.nwb", "sort_group_id": 0, "team": "t"}
    assert deterministic_id("recording", payload) == deterministic_id(
        "recording", payload
    )


def test_numpy_int_matches_plain_int():
    """A numpy ``sort_group_id`` hashes the same as the plain int it
    becomes once stored, so a fetched-then-reinserted id is idempotent."""
    import numpy as np

    assert deterministic_id(
        "recording", {"sort_group_id": np.int64(3)}
    ) == deterministic_id("recording", {"sort_group_id": 3})


def test_int_and_str_int_are_distinct():
    """A bare int and its string form are not conflated; only UUID-like
    strings are normalized."""
    assert deterministic_id(
        "recording", {"sort_group_id": 3}
    ) != deterministic_id("recording", {"sort_group_id": "3"})


def test_no_artifact_does_not_alias_real_artifact():
    """``artifact_detection_id=None`` never aliases a real detection id."""
    u = uuid.uuid4()
    assert deterministic_id(
        "sorting", {"artifact_detection_id": None}
    ) != deterministic_id("sorting", {"artifact_detection_id": u})


def test_kind_namespaces_the_three_tables():
    """Identical payloads in different tables never alias."""
    payload = {"x": 1}
    ids = {
        deterministic_id(k, payload)
        for k in ("recording", "artifact_detection", "sorting")
    }
    assert len(ids) == 3


def test_source_kind_participates_in_identity():
    """A recording-source and a shared-group-source artifact-detection selection
    with otherwise-identical payloads get different ids."""
    base = {"artifact_detection_params_name": "p", "v": 1}
    assert deterministic_id(
        "artifact_detection", {**base, "source_kind": "recording"}
    ) != deterministic_id(
        "artifact_detection", {**base, "source_kind": "shared_artifact_group"}
    )


def test_sort_and_motion_estimate_ids_match_pinned_values():
    """Sort and motion-estimate ids hash the source ids, not the table layout.

    Values computed at 9a50238fd, when the source was stored on part tables.
    """
    from spyglass.spikesorting.v2._core.selection_plan import (
        build_sorting_selection_plan,
    )
    from spyglass.spikesorting.v2._motion.estimation import (
        motion_estimate_identity_payload,
    )

    rec, concat, art, corrected = (uuid.UUID(int=n) for n in (1, 2, 3, 4))
    sorter = {
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_30khz_ms5_2026_06",
    }
    sorts = {
        "e93d48e1-17ea-5d68-9640-d344d4838ef4": {"recording_id": rec},
        "7acb21bd-4a24-535c-b998-469002d00ff1": {
            "recording_id": rec,
            "artifact_detection_id": art,
        },
        "91392c24-04b4-586c-b293-e40fcf514b5d": {"concat_recording_id": concat},
        "4b4d75d8-2e33-5032-b627-c40a4bf6fe75": {
            "recording_id": rec,
            "motion_corrected_recording_id": corrected,
        },
    }
    for pinned, source in sorts.items():
        plan = build_sorting_selection_plan({**source, **sorter})
        assert plan.sorting_id == uuid.UUID(pinned), source

    master = {
        "source_content_hash": "a" * 64,
        "motion_estimation_params_name": "dredge_fast_v1",
        "resolved_params_hash": "b" * 64,
        "spikeinterface_version": "0.104.3",
        "motion_algorithm_version": 1,
    }
    estimates = {
        "082670db-d747-5484-84a9-d05d98b8e9c1": ("recording", rec, art),
        "d7b5ae9d-a6ea-5d59-8d31-2fa4108244cd": (
            "concatenated_recording",
            concat,
            None,
        ),
    }
    for pinned, (kind, source_id, detection) in estimates.items():
        payload = motion_estimate_identity_payload(
            source_kind=kind,
            source_id=source_id,
            artifact_detection_id=detection,
            **master,
        )
        assert deterministic_id("motion_estimate", payload) == uuid.UUID(
            pinned
        ), kind


def test_canonical_json_is_sorted_compact_json():
    out = canonical_json({"b": 2, "a": 1})
    assert out == '{"a":1,"b":2}'


def test_canonical_json_rejects_unsupported_type():
    """A float (or any unsupported type) raises rather than silently
    serializing to a repr-dependent string."""
    with pytest.raises(TypeError, match="UUID, str, int, bool, or None"):
        canonical_json({"x": 1.5})


def test_assert_supplied_id_matches_accepts_none_and_exact():
    target = uuid.uuid4()
    # None -> no-op (the normal path: the helper derives the id).
    assert assert_supplied_id_matches(None, target, field="x") is None
    # exact match (UUID or str) -> no-op.
    assert assert_supplied_id_matches(target, target, field="x") is None
    assert assert_supplied_id_matches(str(target), target, field="x") is None


def test_assert_supplied_id_matches_rejects_mismatch():
    target = uuid.uuid4()
    with pytest.raises(ValueError, match="does not match the deterministic"):
        assert_supplied_id_matches(uuid.uuid4(), target, field="recording_id")


def test_recording_identity_payload_rejects_extra_fields():
    """Joined/fetched dicts must not change the deterministic recording id."""
    key = {
        "nwb_file_name": "session.nwb",
        "sort_group_id": 3,
        "interval_list_name": "raw data valid times",
        "preprocessing_params_name": "default",
        "team_name": "team",
        "analysis_file_name": "joined-extra.nwb",
    }

    with pytest.raises(ValueError, match="unknown field"):
        recording_identity_payload(key)


def test_recording_identity_payload_requires_schema_fields():
    key = {
        "nwb_file_name": "session.nwb",
        "sort_group_id": 3,
        "interval_list_name": "raw data valid times",
        "preprocessing_params_name": "default",
    }

    with pytest.raises(ValueError, match="requires field"):
        recording_identity_payload(key)


def test_recording_identity_payload_plucks_ordered_schema_fields():
    key = {
        "recording_id": uuid.uuid4(),
        "nwb_file_name": "session.nwb",
        "sort_group_id": 3,
        "interval_list_name": "raw data valid times",
        "preprocessing_params_name": "default",
        "team_name": "team",
    }

    payload = recording_identity_payload(key)
    assert tuple(payload) == RECORDING_IDENTITY_FIELDS
    assert "recording_id" not in payload


def test_selection_identity_import_pulls_no_db_layer_modules():
    """A cold import of ``_selection_identity`` pulls in neither
    ``spyglass.common`` nor any v2 *schema* module.

    Those open a DB connection at import; the DataJoint library alone does
    not. A ``spawn`` worker computing an id therefore never connects.
    """
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent("""
        import sys
        import spyglass.spikesorting.v2._core.selection_identity as s
        assert hasattr(s, "deterministic_id")
        leaked = sorted(
            m
            for m in sys.modules
            if m.startswith("spyglass.common")
            or m in (
                "spyglass.spikesorting.v2.recording",
                "spyglass.spikesorting.v2.artifact",
                "spyglass.spikesorting.v2.sorting",
            )
        )
        assert not leaked, "cold import pulled in DB-layer modules: " + repr(leaked)
        """)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "selection-identity helper must import without pulling in the "
        "DB-layer (spyglass.common) or v2 schema modules\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


@pytest.mark.parametrize(
    "errno, message, recoverable",
    [
        (1062, "Duplicate entry 'x' for key 'PRIMARY'", True),
        (1452, "Foreign key constraint fk_1062 fails", False),
        (1451, "Cannot delete or update a parent row", False),
    ],
)
def test_populate_recovery_classifies_translated_errors(
    errno, message, recoverable
):
    """Only translated duplicate errors with an existing row are recovered."""
    import pymysql
    from datajoint.connection import translate_query_error

    from spyglass.spikesorting.v2._orchestration.stages import (
        _populate_once,
    )

    error = translate_query_error(
        pymysql.err.IntegrityError(errno, message), "INSERT ..."
    )

    class Table:
        def populate(self, key, **kwargs):
            raise error

        def __and__(self, key):
            return True

    if recoverable:
        _populate_once(Table(), {"id": 1})
    else:
        with pytest.raises(type(error)) as raised:
            _populate_once(Table(), {"id": 1})
        assert raised.value is error


def test_bool_not_collapsed_to_int_in_identity():
    """``bool`` does not canonicalize to the equivalent ``int``."""
    assert deterministic_id("recording", {"x": True}) != deterministic_id(
        "recording", {"x": 1}
    )
    assert deterministic_id("recording", {"x": False}) != deterministic_id(
        "recording", {"x": 0}
    )


def test_assert_supplied_id_matches_non_uuid_string_gives_curated_error():
    """A non-UUID-parseable supplied id raises the curated "does not match"
    message, not a low-level uuid parse error."""
    target = uuid.uuid4()
    with pytest.raises(ValueError, match="does not match the deterministic"):
        assert_supplied_id_matches("not-a-uuid", target, field="recording_id")


# --------------------------------------------------------------------------
# recording_input_hash: membership, reference and interpolate bad channels.
# Order-independent over the id sets; no DB.
# --------------------------------------------------------------------------


def _rih(**over):
    """Baseline recording_input_hash kwargs, overridable per-case."""
    base = dict(
        electrode_ids=[2, 0, 1],
        reference_mode="none",
        reference_electrode_id=None,
        interpolated_bad_channel_ids=[],
        saved_intervals=[[0.0, 10.0]],
    )
    base.update(over)
    return recording_input_hash(**base)


def test_recording_input_hash_is_stable_hex():
    """Deterministic 64-char sha256 hex, stable across repeated calls."""
    h = _rih()
    assert isinstance(h, str)
    assert len(h) == 64
    assert h == _rih()


def test_recording_input_hash_membership_order_independent():
    """Electrode membership order does not change the hash."""
    assert _rih(electrode_ids=[0, 1, 2]) == _rih(electrode_ids=[2, 1, 0])


def test_recording_input_hash_bad_channel_order_independent():
    """The interpolate bad-channel set is order-independent too."""
    assert _rih(interpolated_bad_channel_ids=[5, 9]) == _rih(
        interpolated_bad_channel_ids=[9, 5]
    )


def test_recording_input_hash_sensitive_to_membership():
    """Adding/removing a sort-group electrode changes the hash."""
    assert _rih(electrode_ids=[0, 1, 2]) != _rih(electrode_ids=[0, 1, 2, 3])


def test_recording_input_hash_sensitive_to_reference():
    """A changed reference mode or electrode changes the hash."""
    assert _rih(reference_mode="none") != _rih(reference_mode="specific")
    assert _rih(reference_mode="specific", reference_electrode_id=4) != _rih(
        reference_mode="specific", reference_electrode_id=7
    )


def test_recording_input_hash_sensitive_to_interpolate_set():
    """A changed live bad-channel set on the interpolate path changes the
    hash; an empty set is distinct from a non-empty one."""
    assert _rih(interpolated_bad_channel_ids=[]) != _rih(
        interpolated_bad_channel_ids=[5]
    )
    assert _rih(interpolated_bad_channel_ids=[5]) != _rih(
        interpolated_bad_channel_ids=[5, 9]
    )


def test_recording_input_hash_numpy_ints_match_plain_ints():
    """A numpy electrode id hashes identically to the plain int it becomes
    once stored, so a fetched-then-rehashed set is idempotent."""
    import numpy as np

    assert _rih(electrode_ids=[np.int64(0), np.int64(1)]) == _rih(
        electrode_ids=[0, 1]
    )
    assert _rih(
        reference_mode="specific", reference_electrode_id=np.int64(3)
    ) == _rih(reference_mode="specific", reference_electrode_id=3)


def test_recording_input_hash_sensitive_to_saved_intervals():
    """Any change in the saved time cut changes the hash."""
    base = _rih(saved_intervals=[[0.0, 10.0]])
    assert base != _rih(saved_intervals=[[0.0, 10.5]])
    assert base != _rih(saved_intervals=[[0.0, 5.0], [6.0, 10.0]])
    assert base != _rih(saved_intervals=[])


def test_recording_input_hash_saved_intervals_array_matches_list():
    """A fetched float64 interval array hashes like the same plain floats."""
    import numpy as np

    assert _rih(saved_intervals=np.array([[0.0, 10.0]])) == _rih(
        saved_intervals=[[0, 10.0]]
    )


def _bypass_message(bypassed):
    return f"bypassed {bypassed}"


def test_existing_selection_pk_none_when_no_master_matches():
    import numpy as np

    canonical = uuid.UUID(int=1)
    for empty in ([], set(), np.array([], dtype=object)):
        assert (
            existing_selection_pk(
                empty,
                canonical,
                pk_field="x_id",
                bypass_message=_bypass_message,
            )
            is None
        )


def test_existing_selection_pk_returns_canonical_master():
    import numpy as np

    canonical = uuid.UUID(int=1)
    for found in ([canonical], {canonical}, np.array([canonical])):
        assert existing_selection_pk(
            found, canonical, pk_field="x_id", bypass_message=_bypass_message
        ) == {"x_id": canonical}


def test_existing_selection_pk_rejects_any_non_deterministic_master():
    """A bypassed id raises even beside the canonical row; the message gets
    the bypassed ids in the order the caller collected them."""
    from spyglass.spikesorting.v2.exceptions import DuplicateSelectionError

    canonical, first, second = (uuid.UUID(int=i) for i in (1, 2, 3))
    # ``second`` is collected before ``first``.
    for found in ([first], [canonical, second, first]):
        with pytest.raises(DuplicateSelectionError) as exc_info:
            existing_selection_pk(
                found,
                canonical,
                pk_field="x_id",
                bypass_message=_bypass_message,
            )
        expected = [mid for mid in found if mid != canonical]
        assert str(exc_info.value) == f"bypassed {expected}"


# --------------------------------------------------------------------------
# DB-tier integration tests.
# --------------------------------------------------------------------------


@pytest.fixture
def fresh_recording_identity(populated_sorting):
    """A RecordingSelection logical identity guaranteed absent from the DB.

    Reuses the populated recording's FK targets with a unique ``team_name``.
    Tears down its rows and team.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection & populated_sorting).fetch1("recording_id")
    rec_row = (RecordingSelection & {"recording_id": rec_id}).fetch1()

    team_name = "v2_selection_identity_test_team"
    LabTeam.insert1(
        {"team_name": team_name, "team_description": "selection-identity test"},
        skip_duplicates=True,
    )
    identity = {
        "nwb_file_name": rec_row["nwb_file_name"],
        "sort_group_id": int(rec_row["sort_group_id"]),
        "interval_list_name": rec_row["interval_list_name"],
        "preprocessing_params_name": rec_row["preprocessing_params_name"],
        "team_name": team_name,
    }
    # Pre-clean in case a crashed prior run left the row.
    (RecordingSelection & identity).delete(safemode=False)
    yield identity
    (RecordingSelection & identity).delete(safemode=False)
    (LabTeam & {"team_name": team_name}).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_recording_selection_writes_deterministic_pk(fresh_recording_identity):
    """``insert_selection`` writes the content-addressed id (FK set + resolved
    input hash) as the PK, stores the hash on the row, and a repeated call
    returns the same id with no second row."""
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        resolve_recording_input_hash,
    )

    identity = fresh_recording_identity
    input_hash = resolve_recording_input_hash(
        identity["nwb_file_name"],
        identity["sort_group_id"],
        identity["preprocessing_params_name"],
        identity["interval_list_name"],
    )
    expected = deterministic_id(
        "recording", {**identity, "recording_input_hash": input_hash}
    )

    pk = RecordingSelection.insert_selection(identity)
    assert pk == {"recording_id": expected}
    assert len(RecordingSelection & identity) == 1
    # The resolved input fingerprint is snapshotted on the row.
    assert (RecordingSelection & pk).fetch1(
        "recording_input_hash"
    ) == input_hash

    # Idempotent: same id, still one row.
    pk_again = RecordingSelection.insert_selection(dict(identity))
    assert pk_again == pk
    assert len(RecordingSelection & identity) == 1


@pytest.mark.slow
@pytest.mark.integration
def test_recording_selection_rejects_mismatched_supplied_id(
    fresh_recording_identity,
):
    """Supplying a random ``recording_id`` that is not the deterministic id
    raises before any row is written."""
    from spyglass.spikesorting.v2.recording import RecordingSelection

    bad = {**fresh_recording_identity, "recording_id": uuid.uuid4()}
    with pytest.raises(ValueError, match="does not match the deterministic"):
        RecordingSelection.insert_selection(bad)
    assert len(RecordingSelection & fresh_recording_identity) == 0


@pytest.mark.slow
@pytest.mark.integration
def test_recording_selection_rejects_extra_fields(fresh_recording_identity):
    """Joined/fetched columns are rejected before UUID derivation or insert."""
    from spyglass.spikesorting.v2.recording import RecordingSelection

    noisy = {
        **fresh_recording_identity,
        "analysis_file_name": "joined-extra.nwb",
    }

    with pytest.raises(ValueError, match="unknown field"):
        RecordingSelection.insert_selection(noisy)
    assert len(RecordingSelection & fresh_recording_identity) == 0


@pytest.mark.slow
@pytest.mark.integration
def test_recording_selection_duplicate_pk_race_refetches(
    fresh_recording_identity,
):
    """The duplicate-PK race loser refetches the winner's row.

    The row exists but the pre-insert lookup is forced to miss once.
    """
    from spyglass.spikesorting.v2.recording import RecordingSelection

    identity = fresh_recording_identity
    pk = RecordingSelection.insert_selection(identity)
    assert len(RecordingSelection & identity) == 1

    with mock.patch.object(
        RecordingSelection,
        "_find_existing_pk",
        side_effect=[None, pk],
    ) as patched:
        raced = RecordingSelection.insert_selection(dict(identity))

    assert raced == pk
    assert patched.call_count == 2
    assert len(RecordingSelection & identity) == 1


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_selection_artifact_detection_vs_none_distinct(
    populated_sorting,
):
    """Sorts with and without artifact detection on the same recording and
    sorter get distinct ids, each matching its derived id."""
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection & populated_sorting).fetch1("recording_id")
    sorter, sorter_params_name = (SortingSelection & populated_sorting).fetch1(
        "sorter", "sorter_params_name"
    )
    artifact_detection_id = SortingSelection.resolve_artifact_detection(
        populated_sorting
    )
    assert (
        artifact_detection_id is not None
    ), "fixture sort should be artifact-detection-backed"

    # The artifact-detection-backed fixture sort's PK is the deterministic id.
    assert populated_sorting["sorting_id"] == deterministic_id(
        "sorting",
        {
            "source_kind": "recording",
            "recording_id": rec_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
            "artifact_detection_id": artifact_detection_id,
        },
    )

    # A no-artifact-detection sort for the same (recording, sorter) is a different id.
    no_art_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
        }
    )
    try:
        assert no_art_pk["sorting_id"] != populated_sorting["sorting_id"]
        assert no_art_pk["sorting_id"] == deterministic_id(
            "sorting",
            {
                "source_kind": "recording",
                "recording_id": rec_id,
                "sorter": sorter,
                "sorter_params_name": sorter_params_name,
                "artifact_detection_id": None,
            },
        )
        # Idempotent re-call returns the same no-artifact-detection id.
        assert (
            SortingSelection.insert_selection(
                {
                    "recording_id": rec_id,
                    "sorter": sorter,
                    "sorter_params_name": sorter_params_name,
                }
            )
            == no_art_pk
        )
    finally:
        (SortingSelection & no_art_pk).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_recording_selection_rejects_single_nondeterministic_row(
    fresh_recording_identity,
):
    """A SINGLE raw-inserted row with a random recording_id for this logical
    identity is rejected, not returned.

    """
    from spyglass.spikesorting.v2.exceptions import DuplicateSelectionError
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        resolve_recording_input_hash,
    )

    identity = fresh_recording_identity
    # Same logical identity (including the input hash), random PK.
    input_hash = resolve_recording_input_hash(
        identity["nwb_file_name"],
        identity["sort_group_id"],
        identity["preprocessing_params_name"],
        identity["interval_list_name"],
    )
    RecordingSelection.insert1(
        {
            **identity,
            "recording_id": uuid.uuid4(),
            "recording_input_hash": input_hash,
        },
        allow_direct_insert=True,
    )

    with pytest.raises(DuplicateSelectionError, match="non-deterministic"):
        RecordingSelection.insert_selection(dict(identity))
    assert len(RecordingSelection & identity) == 1
    # (fixture teardown removes the planted row.)


@pytest.mark.slow
@pytest.mark.integration
def test_make_fetch_rejects_input_hash_drift(fresh_recording_identity):
    """``make_fetch`` raises ``RecordingInputDriftError`` when the live
    sort-group inputs differ from the row's ``recording_input_hash``.

    A wrong stored hash stands in for drift, so the shared sort group is not
    mutated.
    """
    from spyglass.spikesorting.v2.exceptions import RecordingInputDriftError
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )

    identity = fresh_recording_identity
    wrong_hash = "f" * 64  # syntactically valid, cannot match the live inputs
    rid = uuid.uuid4()
    RecordingSelection.insert1(
        {
            **identity,
            "recording_id": rid,
            "recording_input_hash": wrong_hash,
        },
        allow_direct_insert=True,
    )
    with pytest.raises(RecordingInputDriftError, match="fingerprint"):
        Recording().make_fetch({"recording_id": rid})


@pytest.mark.slow
@pytest.mark.integration
def test_changed_sort_interval_mints_a_new_recording(fresh_recording_identity):
    """Changing a sort interval's times in place gives a new recording_id,
    and the old selection's make_fetch rejects the drifted cut."""
    import numpy as np

    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2._orchestration.reporting import (
        audit_recording_input_hashes,
    )
    from spyglass.spikesorting.v2.exceptions import RecordingInputDriftError
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )

    identity = dict(fresh_recording_identity)
    source = (
        IntervalList
        & {
            "nwb_file_name": identity["nwb_file_name"],
            "interval_list_name": identity["interval_list_name"],
        }
    ).fetch1()
    interval = {
        "nwb_file_name": identity["nwb_file_name"],
        "interval_list_name": "v2_selection_identity_changed_interval",
    }
    start, end = float(source["valid_times"][0][0]), float(
        source["valid_times"][-1][1]
    )
    (RecordingSelection & interval).delete_quick()
    (IntervalList & interval).delete_quick()
    IntervalList.insert1(
        {**interval, "valid_times": np.array([[start, end]])},
    )
    identity["interval_list_name"] = interval["interval_list_name"]
    try:
        first = RecordingSelection.insert_selection(dict(identity))
        assert RecordingSelection.insert_selection(dict(identity)) == first
        # A lab edit of the interval in place (cautious_insert update path).
        IntervalList.insert1(
            {
                **interval,
                "valid_times": np.array([[start, start + (end - start) / 2]]),
            },
            update=True,
        )
        second = RecordingSelection.insert_selection(dict(identity))
        assert second != first
        with pytest.raises(RecordingInputDriftError, match="fingerprint"):
            Recording().make_fetch(first)
        stale = audit_recording_input_hashes(interval)
        assert [entry["recording_id"] for entry in stale] == [
            first["recording_id"]
        ]
    finally:
        (RecordingSelection & interval).delete_quick()
        (IntervalList & interval).delete_quick()


@pytest.mark.slow
@pytest.mark.integration
def test_recording_input_hash_matches_make_fetch(populated_sorting):
    """The hash ``insert_selection`` stores equals the one ``make_fetch``
    re-derives for an unchanged recording."""
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        resolve_recording_input_hash,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection & populated_sorting).fetch1("recording_id")
    row = (RecordingSelection & {"recording_id": rec_id}).fetch1()
    stored = row["recording_input_hash"]
    live = resolve_recording_input_hash(
        row["nwb_file_name"],
        row["sort_group_id"],
        row["preprocessing_params_name"],
        row["interval_list_name"],
    )
    assert live == stored


@pytest.mark.slow
@pytest.mark.integration
def test_recording_selection_missing_params_gives_curated_error(
    fresh_recording_identity,
):
    """A missing ``preprocessing_params_name`` yields the
    ``insert_default()`` message, not the input-hash resolver's fetch1
    error."""
    from spyglass.spikesorting.v2.recording import RecordingSelection

    bogus = {
        **fresh_recording_identity,
        "preprocessing_params_name": "does_not_exist_preproc_params",
    }
    with pytest.raises(ValueError, match="insert_default"):
        RecordingSelection.insert_selection(bogus)
    assert len(RecordingSelection & bogus) == 0


@pytest.mark.slow
@pytest.mark.integration
def test_direct_master_insert_rejected_without_flag(fresh_recording_identity):
    """A direct insert into a selection master is rejected with a pointer to
    insert_selection; allow_direct_insert=True is the explicit escape hatch.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.recording import RecordingSelection

    identity = fresh_recording_identity
    row = {
        **identity,
        "recording_id": uuid.uuid4(),
        "recording_input_hash": "0" * 64,
    }
    # No flag -> rejected (before any DB write).
    with pytest.raises(dj.errors.DataJointError, match="is not supported"):
        RecordingSelection.insert1(row)
    assert len(RecordingSelection & identity) == 0
    # Explicit opt-in -> allowed (and cleaned up by the fixture teardown).
    RecordingSelection.insert1(row, allow_direct_insert=True)
    assert len(RecordingSelection & {"recording_id": row["recording_id"]}) == 1


@pytest.mark.usefixtures("dj_conn")
@pytest.mark.parametrize(
    "module, cls_name, extra",
    [
        (
            "spyglass.spikesorting.v2.recording",
            "RecordingSelection",
            {
                "nwb_file_name": "x_.nwb",
                "sort_group_id": 0,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "t",
            },
        ),
        (
            "spyglass.spikesorting.v2.artifact",
            "RecordingArtifactSelection",
            {"artifact_detection_params_name": "none"},
        ),
        (
            "spyglass.spikesorting.v2.artifact",
            "SharedArtifactSelection",
            {"artifact_detection_params_name": "none"},
        ),
        (
            "spyglass.spikesorting.v2.sorting",
            "SortingSelection",
            {"sorter": "mountainsort5", "sorter_params_name": "p"},
        ),
        (
            "spyglass.spikesorting.v2.unit_matching",
            "UnitMatchSelection",
            {"matcher_params_name": "m", "input_set_hash": "0" * 64},
        ),
        (
            "spyglass.spikesorting.v2.session_group",
            "ConcatenatedRecordingSelection",
            {
                "session_group_owner": "o",
                "session_group_name": "g",
                "preprocessing_params_name": "p",
            },
        ),
    ],
)
def test_direct_master_insert_rejected_all_masters(module, cls_name, extra):
    """Every selection master rejects a no-flag direct insert, before any
    DB write.
    """
    import importlib
    import uuid

    import datajoint as dj

    cls = getattr(importlib.import_module(module), cls_name)
    pk_field = cls.primary_key[0]
    row = {pk_field: uuid.uuid4(), **extra}
    with pytest.raises(dj.errors.DataJointError, match="is not supported"):
        cls.insert1(row)
    assert len(cls & {pk_field: row[pk_field]}) == 0


@pytest.mark.usefixtures("dj_conn")
@pytest.mark.parametrize(
    "module, cls_name",
    [
        ("spyglass.spikesorting.v2.sorting", "SortingSelection"),
        ("spyglass.spikesorting.v2.motion", "MotionEstimateSelection"),
    ],
)
@pytest.mark.parametrize("n_sources", [0, 2])
def test_direct_insert_needs_exactly_one_source(module, cls_name, n_sources):
    """Even with ``allow_direct_insert=True``, a row naming zero or both of
    ``recording_id`` / ``concat_recording_id`` is refused before any write.
    """
    import importlib

    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    cls = getattr(importlib.import_module(module), cls_name)
    pk_field = cls.primary_key[0]
    row = {pk_field: uuid.uuid4()}
    if n_sources == 2:
        row.update(recording_id=uuid.uuid4(), concat_recording_id=uuid.uuid4())
    with pytest.raises(SchemaBypassError, match="exactly one must be set"):
        cls.insert1(row, allow_direct_insert=True)
    assert len(cls & {pk_field: row[pk_field]}) == 0


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_selection_duplicate_pk_race_refetches(populated_sorting):
    """SortingSelection duplicate-PK race loser refetches the winner's row."""
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection & populated_sorting).fetch1("recording_id")
    sorter, sorter_params_name = (SortingSelection & populated_sorting).fetch1(
        "sorter", "sorter_params_name"
    )
    artifact_detection_id = SortingSelection.resolve_artifact_detection(
        populated_sorting
    )
    key = {
        "recording_id": rec_id,
        "sorter": sorter,
        "sorter_params_name": sorter_params_name,
        "artifact_detection_id": artifact_detection_id,
    }
    pk = SortingSelection.insert_selection(key)
    assert pk == populated_sorting

    with mock.patch.object(
        SortingSelection, "_find_existing_pk", side_effect=[None, pk]
    ) as patched:
        raced = SortingSelection.insert_selection(dict(key))

    assert raced == pk
    assert patched.call_count == 2
    assert len(SortingSelection & pk) == 1


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_selection_mismatched_row_raises_schema_bypass(
    populated_sorting,
):
    """A row with the deterministic ``sorting_id`` that stores another
    artifact detection (a raw insert) surfaces a SchemaBypassError on the
    duplicate-PK refetch."""
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection & populated_sorting).fetch1("recording_id")
    sorter, sorter_params_name = (SortingSelection & populated_sorting).fetch1(
        "sorter", "sorter_params_name"
    )
    # A no-artifact sort for this (recording, sorter) has no real master.
    det_id = deterministic_id(
        "sorting",
        {
            "source_kind": "recording",
            "recording_id": rec_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
            "artifact_detection_id": None,
        },
    )
    other_merge_id = (SortingSelection & populated_sorting).fetch1(
        "artifact_detection_merge_id"
    )
    assert other_merge_id is not None
    SortingSelection.insert1(
        {
            "sorting_id": det_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
            "recording_id": rec_id,
            "artifact_detection_merge_id": other_merge_id,
        },
        allow_direct_insert=True,
    )
    try:
        with pytest.raises(SchemaBypassError, match="does not match this"):
            SortingSelection.insert_selection(
                {
                    "recording_id": rec_id,
                    "sorter": sorter,
                    "sorter_params_name": sorter_params_name,
                }
            )
    finally:
        (SortingSelection & {"sorting_id": det_id}).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_artifact_detection_selection_shared_group_writes_deterministic_pk(
    populated_sorting,
):
    """``SharedArtifactSelection.insert_selection`` writes the
    deterministic PK over session, group name and member set, freezes the
    members, is idempotent, and differs from the recording-source id for the
    same params and recording."""
    from spyglass.spikesorting.v2.artifact import (
        SharedArtifactGroup,
        SharedArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection & populated_sorting).fetch1("recording_id")
    nwb_file_name = (RecordingSelection & {"recording_id": rec_id}).fetch1(
        "nwb_file_name"
    )
    group_key = {
        "nwb_file_name": nwb_file_name,
        "shared_artifact_group_name": "v2_selection_identity_shared_group",
    }
    # Single-member group: trivially one-session and time-axis-consistent.
    (SharedArtifactGroup & group_key).delete(safemode=False)
    SharedArtifactGroup.create_group(
        nwb_file_name,
        group_key["shared_artifact_group_name"],
        [{"recording_id": rec_id}],
    )

    expected = deterministic_id(
        "artifact_detection",
        {
            "source_kind": "shared_artifact_group",
            "artifact_detection_params_name": "default",
            **group_key,
            "member_set_hash": sha256_json(
                [str(rec_id)], separators=(", ", ": ")
            ),
        },
    )
    select_key = {**group_key, "artifact_detection_params_name": "default"}
    try:
        pk = SharedArtifactSelection.insert_selection(dict(select_key))
        assert pk == {"artifact_detection_id": expected}
        assert (SharedArtifactSelection.Member & pk).fetch1(
            "recording_id"
        ) == rec_id
        # Idempotent re-call returns the same id.
        assert SharedArtifactSelection.insert_selection(dict(select_key)) == pk
        rec_source_id = deterministic_id(
            "artifact_detection",
            {
                "source_kind": "recording",
                "artifact_detection_params_name": "default",
                "recording_id": rec_id,
            },
        )
        assert expected != rec_source_id
    finally:
        (SharedArtifactGroup & group_key).delete(safemode=False)


def test_shared_artifact_identity_covers_session_and_members():
    """The same group name in another session, or with other members, is a
    different detection; member order is not part of the identity."""
    from spyglass.spikesorting.v2._core.selection_identity import (
        artifact_detection_identity_payload,
    )

    a, b = str(uuid.uuid4()), str(uuid.uuid4())

    def _id(nwb_file_name, members):
        return deterministic_id(
            "artifact_detection",
            artifact_detection_identity_payload(
                artifact_detection_params_name="default",
                shared_artifact_group_name="default",
                nwb_file_name=nwb_file_name,
                member_recording_ids=members,
            ),
        )

    base = _id("s1_.nwb", [a, b])
    assert _id("s1_.nwb", [b, a]) == base
    assert _id("s2_.nwb", [a, b]) != base
    assert _id("s1_.nwb", [a]) != base
    for nwb_file_name, members in ((None, [a]), ("s1_.nwb", [])):
        with pytest.raises(ValueError, match="nwb_file_name and member"):
            _id(nwb_file_name, members)


# --------------------------------------------------------------------------
# In-place ``update1`` is rejected on every selection master and on the
# ``CurationV2`` / ``SessionGroup`` factory masters, before any DB write.
# --------------------------------------------------------------------------

_MASTER_UPDATE1_CASES = [
    ("spyglass.spikesorting.v2.recording", "RecordingSelection"),
    ("spyglass.spikesorting.v2.artifact", "RecordingArtifactSelection"),
    ("spyglass.spikesorting.v2.artifact", "SharedArtifactSelection"),
    ("spyglass.spikesorting.v2.sorting", "SortingSelection"),
    ("spyglass.spikesorting.v2.unit_matching", "UnitMatchSelection"),
    (
        "spyglass.spikesorting.v2.session_group",
        "ConcatenatedRecordingSelection",
    ),
    ("spyglass.spikesorting.v2.metric_curation", "CurationEvaluationSelection"),
    ("spyglass.spikesorting.v2.curation", "CurationV2"),
    ("spyglass.spikesorting.v2.session_group", "SessionGroup"),
    ("spyglass.spikesorting.v2.artifact", "SharedArtifactGroup"),
]


def _synthetic_master_pk(cls):
    """Build a syntactically valid PK dict for a master (no DB row needed).

    The guard rejects before any lookup, so the values only fill the PK.
    """
    pk = {}
    for field in cls.primary_key:
        attr = cls.heading.attributes[field]
        if attr.type == "uuid":
            pk[field] = uuid.uuid4()
        elif attr.numeric:
            pk[field] = 0
        else:
            pk[field] = "synthetic_audit_probe"
    return pk


@pytest.mark.usefixtures("dj_conn")
@pytest.mark.parametrize("module, cls_name", _MASTER_UPDATE1_CASES)
def test_update1_rejected_all_masters(module, cls_name):
    """In-place ``update1`` is rejected on every selection master and on the
    ``CurationV2`` / ``SessionGroup`` factory masters; the ``allow_master_mutation``
    escape hatch forwards to the real ``update1``."""
    import importlib

    import datajoint as dj

    cls = getattr(importlib.import_module(module), cls_name)
    row = _synthetic_master_pk(cls)
    with pytest.raises(dj.errors.DataJointError, match="is not supported"):
        cls().update1(row)


@pytest.mark.usefixtures("dj_conn")
@pytest.mark.parametrize(
    "module, cls_name",
    [
        ("spyglass.spikesorting.v2.curation", "CurationV2"),
        ("spyglass.spikesorting.v2.session_group", "SessionGroup"),
        ("spyglass.spikesorting.v2.artifact", "SharedArtifactGroup"),
    ],
)
def test_factory_master_direct_insert_rejected(module, cls_name):
    """A direct ``insert1`` into ``CurationV2``, ``SessionGroup`` or
    ``SharedArtifactGroup`` is rejected; use ``insert_curation`` /
    ``create_group``."""
    import importlib

    import datajoint as dj

    cls = getattr(importlib.import_module(module), cls_name)
    row = _synthetic_master_pk(cls)
    with pytest.raises(dj.errors.DataJointError, match="is not supported"):
        cls().insert1(row)


@pytest.mark.slow
@pytest.mark.integration
def test_master_update1_bypass_allows_mutation(fresh_recording_identity):
    """``allow_master_mutation=True`` forwards to the real ``update1`` for
    a row with no live references."""
    import datajoint as dj

    from spyglass.common import LabTeam
    from spyglass.spikesorting.v2.recording import RecordingSelection

    pk = RecordingSelection.insert_selection(fresh_recording_identity)
    # Without the flag: rejected, row unchanged.
    with pytest.raises(dj.errors.DataJointError, match="is not supported"):
        RecordingSelection().update1({**pk, "team_name": "other_team"})
    current_team = (RecordingSelection & pk).fetch1("team_name")
    replacement_team = {"team_name": "v2_identity_mutation_test_team"}
    assert replacement_team["team_name"] != current_team
    LabTeam.insert1(replacement_team)
    try:
        RecordingSelection().update1(
            {**pk, **replacement_team}, allow_master_mutation=True
        )
        assert (RecordingSelection & pk).fetch1(
            "team_name"
        ) == replacement_team["team_name"]
    finally:
        RecordingSelection().update1(
            {**pk, "team_name": current_team}, allow_master_mutation=True
        )
        (LabTeam & replacement_team).delete(safemode=False)


def test_recording_artifact_detection_ids_are_pinned():
    """Single-recording detection ids keep their base-commit values.

    Captured from ``RecordingArtifactSelection.insert_selection`` on the
    chronic minirec recordings; regenerate (do not loosen) only when the
    recording-source identity is meant to change.
    """
    from spyglass.spikesorting.v2._core.selection_identity import (
        artifact_detection_identity_payload,
    )

    cases = [
        (
            "02a9f450-ff62-5ca3-a184-1dd0554cfa42",
            None,
            "94dc697a-58f8-5054-834d-a72e6fd3cb23",
        ),
        (
            "02a9f450-ff62-5ca3-a184-1dd0554cfa42",
            [[3.0, 3.5]],
            "a07ce75e-58dc-5575-b1ad-02fc3b1a381c",
        ),
        (
            "500e5fd7-cba1-5a77-8146-66701a427228",
            None,
            "bf3f06bb-4627-5de8-bf40-2bb78c8ded09",
        ),
        (
            "500e5fd7-cba1-5a77-8146-66701a427228",
            [[3.0, 3.5]],
            "cab4f7af-2b43-5ab3-93d4-e9be7c6622b6",
        ),
    ]
    for recording_id, exclusions, expected in cases:
        payload = artifact_detection_identity_payload(
            artifact_detection_params_name="default",
            recording_id=uuid.UUID(recording_id),
            manual_excluded_times=exclusions,
        )
        assert deterministic_id("artifact_detection", payload) == uuid.UUID(
            expected
        )
