"""Phase A: deterministic, content-addressed selection identities.

Two layers:

* **Pure-helper unit tests** (no DB, no Docker) pin the canonicalization
  footgun the ``_selection_identity`` module exists to kill -- a ``str``
  vs ``uuid.UUID`` ``artifact_id`` must hash to the SAME id, a numpy
  ``sort_group_id`` must match the plain int, "no artifact" must never
  alias a real id, and the table ``kind`` + ``source_kind`` must
  participate so the three selection tables (and the artifact/no-artifact
  sorts) never collide.
* **DB-tier integration tests** (``slow`` + ``integration``) prove the
  deterministic id is the actual primary key written by
  ``insert_selection``, that repeated calls are idempotent, that a
  caller-supplied mismatched id is rejected, that the duplicate-PK race
  loser refetches the winner's row instead of raising, and that an
  artifact-backed and an artifact-free sort for the same (recording,
  sorter) get distinct ids.
"""

from __future__ import annotations

import uuid
from unittest import mock

import pytest

from spyglass.spikesorting.v2._selection_identity import (
    V2_SELECTION_NAMESPACE,
    assert_supplied_id_matches,
    canonical_identity,
    deterministic_id,
)

# --------------------------------------------------------------------------
# Pure-helper unit tests -- no database, run without Docker.
# --------------------------------------------------------------------------


def test_namespace_is_frozen_literal():
    """The v2 namespace is the frozen uuid5(DNS, seed) literal.

    Pinning the value guards against an accidental edit to the seed string
    silently re-homing every selection id (which would orphan all
    previously-computed ids).
    """
    assert V2_SELECTION_NAMESPACE == uuid.uuid5(
        uuid.NAMESPACE_DNS, "spyglass.spikesorting.v2.selection"
    )
    assert str(V2_SELECTION_NAMESPACE) == "b44d4765-4714-5c69-96d5-97feb2217e86"


def test_deterministic_id_returns_uuid():
    assert isinstance(deterministic_id("recording", {"a": 1}), uuid.UUID)


def test_deterministic_id_stable_across_uuid_vs_str():
    """A ``str`` and a ``uuid.UUID`` of the same value hash identically.

    This is the exact bug behind
    ``test_insert_selection_dedup_accepts_str_artifact_id`` -- a str
    ``artifact_id`` once forked a duplicate sort.
    """
    u = uuid.uuid4()
    assert deterministic_id("sorting", {"artifact_id": u}) == deterministic_id(
        "sorting", {"artifact_id": str(u)}
    )


def test_deterministic_id_stable_across_uuid_case():
    """Upper- and lower-case UUID strings canonicalize to one id."""
    u = uuid.uuid4()
    assert deterministic_id(
        "sorting", {"artifact_id": str(u).upper()}
    ) == deterministic_id("sorting", {"artifact_id": str(u)})


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
    np = pytest.importorskip("numpy")
    assert deterministic_id(
        "recording", {"sort_group_id": np.int64(3)}
    ) == deterministic_id("recording", {"sort_group_id": 3})


def test_int_and_str_int_are_distinct():
    """A bare int and its string form are NOT conflated (only UUID-ish
    strings are normalized); the table stores ``sort_group_id`` as an int,
    so callers pass an int and this distinction never bites in practice."""
    assert deterministic_id(
        "recording", {"sort_group_id": 3}
    ) != deterministic_id("recording", {"sort_group_id": "3"})


def test_no_artifact_does_not_alias_real_artifact():
    """``artifact_id=None`` (the single "no artifact pass" form) never
    collides with any real artifact id."""
    u = uuid.uuid4()
    assert deterministic_id(
        "sorting", {"artifact_id": None}
    ) != deterministic_id("sorting", {"artifact_id": u})


def test_kind_namespaces_the_three_tables():
    """Identical payloads in different tables never alias."""
    payload = {"x": 1}
    ids = {
        deterministic_id(k, payload)
        for k in ("recording", "artifact", "sorting")
    }
    assert len(ids) == 3


def test_source_kind_participates_in_identity():
    """A recording-source and a shared-group-source artifact selection
    with otherwise-identical payloads get different ids."""
    base = {"artifact_params_name": "p", "v": 1}
    assert deterministic_id(
        "artifact", {**base, "source_kind": "recording"}
    ) != deterministic_id(
        "artifact", {**base, "source_kind": "shared_artifact_group"}
    )


def test_canonical_identity_is_sorted_compact_json():
    out = canonical_identity({"b": 2, "a": 1})
    assert out == '{"a":1,"b":2}'


def test_canonical_identity_rejects_unsupported_type():
    """A float (or any unsupported type) raises rather than silently
    serializing to a repr-dependent string."""
    with pytest.raises(TypeError, match="UUID, str, int, bool, or None"):
        canonical_identity({"x": 1.5})


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


def test_selection_identity_module_imports_without_db():
    """Regression guard (mirrors the ``_artifact_compute`` kernel guard):
    a cold import of ``_selection_identity`` in a fresh interpreter pulls
    in neither ``spyglass.common`` nor any v2 *schema* module, so an HPC
    job array that computes a selection id in a spawned worker never opens
    a DB connection at import.
    """
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent("""
        import sys
        import spyglass.spikesorting.v2._selection_identity as s
        assert hasattr(s, "deterministic_id")
        assert hasattr(s, "canonical_identity")
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
        "selection-identity helper must import without a DB dependency\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# --------------------------------------------------------------------------
# DB-tier integration tests -- need the Docker MySQL + populated fixture.
# --------------------------------------------------------------------------


@pytest.fixture
def fresh_recording_identity(populated_sorting):
    """A RecordingSelection logical identity guaranteed absent from the DB.

    Reuses the populated recording's existing FK targets (Raw,
    SortGroupV2, IntervalList, PreprocessingParameters) but a UNIQUE
    ``team_name`` so no prior row -- even one left by an earlier run --
    can exist for this exact identity. That lets the determinism /
    race assertions below check ``insert_selection`` from a known-empty
    starting point. Tears down the row(s) and the team it created.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection.RecordingSource & populated_sorting).fetch1(
        "recording_id"
    )
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
        "preproc_params_name": rec_row["preproc_params_name"],
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
    """``insert_selection`` writes the content-addressed id as the PK, and
    a repeated call returns the same id with no second row."""
    from spyglass.spikesorting.v2.recording import RecordingSelection

    identity = fresh_recording_identity
    expected = deterministic_id("recording", identity)

    pk = RecordingSelection.insert_selection(identity)
    assert pk == {"recording_id": expected}
    assert len(RecordingSelection & identity) == 1

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
def test_recording_selection_duplicate_pk_race_refetches(
    fresh_recording_identity,
):
    """The duplicate-PK race loser refetches the winner's row.

    Simulate the TOCTOU window deterministically: the row already exists,
    but the pre-insert lookup is forced to "miss" once (as if a concurrent
    caller inserted between our check and our insert). The insert then
    hits the deterministic-PK uniqueness constraint, the narrowed
    duplicate-key catch fires, and the refetch returns the existing row --
    no new row, no error.
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
    # The pre-check missed, the insert collided, the refetch returned it.
    assert patched.call_count == 2
    # No duplicate row was created by the losing insert.
    assert len(RecordingSelection & identity) == 1


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_selection_artifact_vs_no_artifact_distinct(populated_sorting):
    """An artifact-backed and an artifact-free sort for the same
    (recording, sorter) get DISTINCT deterministic ids, and each matches
    the id computed directly from its logical identity."""
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection.RecordingSource & populated_sorting).fetch1(
        "recording_id"
    )
    sorter, sorter_params_name = (SortingSelection & populated_sorting).fetch1(
        "sorter", "sorter_params_name"
    )
    artifact_id = SortingSelection.resolve_artifact(populated_sorting)
    assert artifact_id is not None, "fixture sort should be artifact-backed"

    # The artifact-backed fixture sort's PK is the deterministic id.
    assert populated_sorting["sorting_id"] == deterministic_id(
        "sorting",
        {
            "source_kind": "recording",
            "recording_id": rec_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
            "artifact_id": artifact_id,
        },
    )

    # A no-artifact sort for the same (recording, sorter) is a different id.
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
                "artifact_id": None,
            },
        )
        # Idempotent re-call returns the same no-artifact id.
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
