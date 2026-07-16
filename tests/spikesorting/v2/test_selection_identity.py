"""Deterministic, content-addressed selection identities.

Two layers:

* **Pure-helper unit tests** (no DB, no Docker) pin the canonicalization
  footgun the ``_selection_identity`` module exists to kill -- a ``str``
  vs ``uuid.UUID`` ``artifact_detection_id`` must hash to the SAME id, a numpy
  ``sort_group_id`` must match the plain int, "no artifact-detection pass"
  must never alias a real id, and the table ``kind`` + ``source_kind`` must
  participate so the three selection tables (and the
  artifact-detection/no-artifact-detection sorts) never collide.
* **DB-tier integration tests** (``slow`` + ``integration``) prove the
  deterministic id is the actual primary key written by
  ``insert_selection``, that repeated calls are idempotent, that a
  caller-supplied mismatched id is rejected, that the duplicate-PK race
  loser refetches the winner's row instead of raising, and that an
  artifact-detection-backed and an artifact-detection-free sort for the same
  (recording, sorter) get distinct ids.
"""

from __future__ import annotations

import uuid
from unittest import mock

import pytest

from spyglass.spikesorting.v2._selection_identity import (
    RECORDING_IDENTITY_FIELDS,
    V2_SELECTION_NAMESPACE,
    assert_supplied_id_matches,
    canonical_identity,
    deterministic_id,
    recording_identity_payload,
    recording_input_hash,
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
    ``test_insert_selection_dedup_accepts_str_artifact_detection_id`` -- a str
    ``artifact_detection_id`` once forked a duplicate sort.
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

    Those are the modules that open a DB connection at import (``dj.schema``
    activation / ``from spyglass.common import ...``); the leaf module
    itself imports only ``json`` + ``uuid``. This is the precise,
    checkable claim -- it does NOT assert "no datajoint anywhere", because
    importing the top-level ``spyglass`` package (via ``settings``)
    legitimately imports the DataJoint *library*, which does not connect on
    its own. The guarantee that matters: an HPC job array computing a
    selection id in a ``spawn`` worker re-imports only this DB-free leaf,
    so it never opens a connection. Mirrors the ``_artifact_compute``
    kernel guard.
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
        "selection-identity helper must import without pulling in the "
        "DB-layer (spyglass.common) or v2 schema modules\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# Each case builds its exception lazily inside the test (where ``datajoint``
# is imported) from ``(error_name, args)`` so the parametrize decorator stays
# pure data and the module-under-test's import boundary is unaffected.
@pytest.mark.parametrize(
    "error_name, args, expected",
    [
        # DataJoint maps MySQL errno 1062 -> DuplicateError: that is the race.
        pytest.param(
            "DuplicateError", ("Duplicate entry",), True, id="duplicate-error"
        ),
        # FK violations surface as IntegrityError with no 1062 errno -- they
        # MUST propagate, not be swallowed.
        pytest.param(
            "IntegrityError",
            (
                "Cannot add or update a child row: a foreign key constraint "
                "fails",
            ),
            False,
            id="fk-integrity-error",
        ),
        # Defensive: a raw, untranslated connector error carrying the
        # structured errno 1062 as the first arg is recognized.
        pytest.param(
            "IntegrityError",
            (1062, "Duplicate entry 'x' for key 'PRIMARY'"),
            True,
            id="structured-errno-1062",
        ),
        # Rendered database messages alone are deliberately ignored: they vary
        # by connector/version/locale and are not safe enough to recover from.
        pytest.param(
            "IntegrityError",
            ("Duplicate entry 'x' for key 'PRIMARY'",),
            False,
            id="rendered-duplicate-message-only",
        ),
        # CRITICAL false-positive guard: an FK-violation IntegrityError whose
        # message merely CONTAINS the digits "1062" (in a constraint name, a
        # rendered UUID) must NOT be treated as a duplicate -- a bare "1062"
        # substring match would silently swallow a real FK error.
        pytest.param(
            "IntegrityError",
            (
                "a foreign key constraint fails (`db`.`t`, "
                "CONSTRAINT `fk_1062`)",
            ),
            False,
            id="fk-error-containing-1062-substring",
        ),
    ],
)
def test_is_duplicate_key_error_classifies_datajoint_exceptions(
    error_name, args, expected
):
    """``_is_duplicate_key_error`` is True only for a duplicate PRIMARY-KEY
    violation, never for an FK / missing-source-part ``IntegrityError``."""
    import datajoint as dj

    from spyglass.spikesorting.v2.utils import _is_duplicate_key_error

    error = getattr(dj.errors, error_name)(*args)
    assert _is_duplicate_key_error(error) is expected


def test_is_duplicate_key_error_ignores_unrelated_exception():
    """An unrelated exception is not a duplicate-key error."""
    from spyglass.spikesorting.v2.utils import _is_duplicate_key_error

    assert not _is_duplicate_key_error(ValueError("unrelated"))


def test_bool_not_collapsed_to_int_in_identity():
    """``bool`` must not canonicalize to the equivalent ``int`` (the
    bool-before-int branch ordering in _canonical_scalar is load-bearing)."""
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
# recording_input_hash -- content-address a recording's RESOLVED construction
# inputs (membership + reference + interpolate bad-channel set) so a changed
# input mints a new recording_id. Order-independent over the two id sets;
# sensitive to every field. Pure + DB-free (fed already-resolved values).
# --------------------------------------------------------------------------


def _rih(**over):
    """Baseline recording_input_hash kwargs, overridable per-case."""
    base = dict(
        electrode_ids=[2, 0, 1],
        reference_mode="none",
        reference_electrode_id=None,
        interpolated_bad_channel_ids=[],
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
    """Electrode membership is a SET: a re-query in a different row order
    yields the same hash (the ids are sorted before hashing)."""
    assert _rih(electrode_ids=[0, 1, 2]) == _rih(electrode_ids=[2, 1, 0])


def test_recording_input_hash_bad_channel_order_independent():
    """The interpolate bad-channel set is order-independent too."""
    assert _rih(interpolated_bad_channel_ids=[5, 9]) == _rih(
        interpolated_bad_channel_ids=[9, 5]
    )


def test_recording_input_hash_sensitive_to_membership():
    """Adding/removing an electrode (the OP-4 drift) changes the hash."""
    assert _rih(electrode_ids=[0, 1, 2]) != _rih(electrode_ids=[0, 1, 2, 3])


def test_recording_input_hash_sensitive_to_reference():
    """A changed reference mode or electrode changes the hash."""
    assert _rih(reference_mode="none") != _rih(reference_mode="specific")
    assert _rih(reference_mode="specific", reference_electrode_id=4) != _rih(
        reference_mode="specific", reference_electrode_id=7
    )


def test_recording_input_hash_sensitive_to_interpolate_set():
    """A changed interpolate bad-channel set (the OP-3 drift) changes the
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
    np = pytest.importorskip("numpy")
    assert _rih(electrode_ids=[np.int64(0), np.int64(1)]) == _rih(
        electrode_ids=[0, 1]
    )
    assert _rih(
        reference_mode="specific", reference_electrode_id=np.int64(3)
    ) == _rih(reference_mode="specific", reference_electrode_id=3)


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
def test_sorting_selection_artifact_detection_vs_none_distinct(
    populated_sorting,
):
    """An artifact-detection-backed and no-artifact-detection sort for the same
    (recording, sorter) get DISTINCT deterministic ids, and each matches
    the id computed directly from its logical identity."""
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection.RecordingSource & populated_sorting).fetch1(
        "recording_id"
    )
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

    The invariant is that the logical identity maps to ONE
    content-addressed id. A pre-determinism / raw-insert row with a
    different (random) recording_id violates that, so insert_selection
    raises instead of silently adopting it as canonical.
    """
    from spyglass.spikesorting.v2.exceptions import DuplicateSelectionError
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        resolve_recording_input_hash,
    )

    identity = fresh_recording_identity
    # One raw row with the SAME logical identity (FK set + resolved input hash)
    # but a random PK. The hash must match so the row shares the logical
    # identity find-existing scopes to; without it the raw row would sit under a
    # different identity axis and go undetected. The master insert guard requires
    # allow_direct_insert=True for this deliberate bypass (all FK targets exist,
    # so no FK-checks-off needed).
    input_hash = resolve_recording_input_hash(
        identity["nwb_file_name"],
        identity["sort_group_id"],
        identity["preprocessing_params_name"],
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
    # The helper did NOT converge a second (deterministic) row on top.
    assert len(RecordingSelection & identity) == 1
    # (fixture teardown removes the planted row.)


@pytest.mark.slow
@pytest.mark.integration
def test_make_fetch_rejects_input_hash_drift(fresh_recording_identity):
    """``make_fetch`` raises ``RecordingInputDriftError`` when the live
    sort-group inputs no longer match the ``recording_input_hash`` the row
    carries -- the membership/reference/bad-channel set changed after the id was
    minted. Exercised by planting a row with a deliberately wrong fingerprint
    (a real drift produces the same mismatch), so the guard is proven without
    mutating the shared sort group."""
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
def test_recording_input_hash_matches_make_fetch(populated_sorting):
    """The hash ``insert_selection`` stored equals the one ``make_fetch``
    re-derives for the SAME unchanged recording, so a normal populate never
    false-drifts. Guards the ``resolve_recording_input_hash`` <-> ``make_fetch``
    lock-step: if their input-reading logic diverged, this equality would
    break (and a real recording would raise a spurious drift)."""
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        resolve_recording_input_hash,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection.RecordingSource & populated_sorting).fetch1(
        "recording_id"
    )
    row = (RecordingSelection & {"recording_id": rec_id}).fetch1()
    stored = row["recording_input_hash"]
    # The pipeline created this recording via insert_selection, so it carries a
    # fingerprint (never NULL).
    assert stored is not None
    live = resolve_recording_input_hash(
        row["nwb_file_name"],
        row["sort_group_id"],
        row["preprocessing_params_name"],
    )
    assert live == stored


@pytest.mark.slow
@pytest.mark.integration
def test_recording_selection_missing_params_gives_curated_error(
    fresh_recording_identity,
):
    """A missing ``preprocessing_params_name`` yields the curated
    ``insert_default()`` message, not the raw empty-fetch error from the
    input-hash resolver (which ``fetch1``'s that params row). The curated
    lookup check must run BEFORE the resolver."""
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

    (RecordingSelection covers the reject + happy-path pair with real FKs;
    test_direct_master_insert_rejected_all_masters covers the reject path for
    all three masters.)
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.recording import RecordingSelection

    identity = fresh_recording_identity
    row = {**identity, "recording_id": uuid.uuid4()}
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
            "ArtifactDetectionSelection",
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
            {
                "session_group_owner": "o",
                "session_group_name": "g",
                "matcher_params_name": "m",
                "curation_set_hash": "0" * 64,
            },
        ),
        (
            "spyglass.spikesorting.v2.session_group",
            "ConcatenatedRecordingSelection",
            {
                "session_group_owner": "o",
                "session_group_name": "g",
                "preprocessing_params_name": "p",
                "motion_correction_params_name": "m",
            },
        ),
    ],
)
def test_direct_master_insert_rejected_all_masters(module, cls_name, extra):
    """All FIVE selection masters reject a no-flag direct insert. The guard
    is a shared mixin, but a per-class regression (a class dropping the mixin
    from its MRO, or a future per-class insert override that forgets to
    forward allow_direct_insert) must surface. The guard raises BEFORE any DB
    write, so the bogus-FK row never reaches MySQL.
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


@pytest.mark.slow
@pytest.mark.integration
def test_artifact_detection_selection_duplicate_pk_race_refetches(
    populated_sorting,
):
    """ArtifactDetectionSelection duplicate-PK race loser refetches the winner's
    master+source row (no new row, no error)."""
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection.RecordingSource & populated_sorting).fetch1(
        "recording_id"
    )
    key = {"recording_id": rec_id, "artifact_detection_params_name": "none"}
    pk = ArtifactDetectionSelection.insert_selection(
        key
    )  # the populated selection

    with mock.patch.object(
        ArtifactDetectionSelection, "_find_existing_pk", side_effect=[None, pk]
    ) as patched:
        raced = ArtifactDetectionSelection.insert_selection(dict(key))

    assert raced == pk
    assert patched.call_count == 2
    assert (
        len(
            ArtifactDetectionSelection
            & {"artifact_detection_id": pk["artifact_detection_id"]}
        )
        == 1
    )


@pytest.mark.slow
@pytest.mark.integration
def test_artifact_detection_selection_orphan_master_raises_schema_bypass(
    populated_sorting,
):
    """A deterministic ArtifactDetectionSelection master with NO source part (a
    raw-insert orphan) surfaces a SchemaBypassError on the duplicate-PK
    refetch, not the opaque duplicate-key error."""
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionSelection
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection.RecordingSource & populated_sorting).fetch1(
        "recording_id"
    )
    # "default" params exist (insert_default) but are unused by the fixture,
    # so this (recording, "default") identity has no real master yet.
    det_id = deterministic_id(
        "artifact_detection",
        {
            "source_kind": "recording",
            "artifact_detection_params_name": "default",
            "recording_id": rec_id,
        },
    )
    ArtifactDetectionSelection.insert1(
        {
            "artifact_detection_id": det_id,
            "artifact_detection_params_name": "default",
        },
        allow_direct_insert=True,
    )
    try:
        with pytest.raises(SchemaBypassError, match="raw-insert orphan"):
            ArtifactDetectionSelection.insert_selection(
                {
                    "recording_id": rec_id,
                    "artifact_detection_params_name": "default",
                }
            )
    finally:
        (ArtifactDetectionSelection & {"artifact_detection_id": det_id}).delete(
            safemode=False
        )


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_selection_duplicate_pk_race_refetches(populated_sorting):
    """SortingSelection duplicate-PK race loser refetches the winner's row."""
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection.RecordingSource & populated_sorting).fetch1(
        "recording_id"
    )
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
def test_sorting_selection_orphan_master_raises_schema_bypass(
    populated_sorting,
):
    """A deterministic SortingSelection master with NO recording source
    part (a raw-insert orphan) surfaces a SchemaBypassError on the
    duplicate-PK refetch."""
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection.RecordingSource & populated_sorting).fetch1(
        "recording_id"
    )
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
    SortingSelection.insert1(
        {
            "sorting_id": det_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
        },
        allow_direct_insert=True,
    )
    try:
        with pytest.raises(SchemaBypassError, match="raw-insert orphan"):
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
    """The shared-group (cross-recording) source branch of
    ArtifactDetectionSelection.insert_selection also writes the deterministic PK,
    is idempotent, and does NOT alias the recording-source id for the same
    params + recording (source_kind disambiguates)."""
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionSelection,
        SharedArtifactGroup,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_id = (SortingSelection.RecordingSource & populated_sorting).fetch1(
        "recording_id"
    )
    group_name = "v2_selection_identity_shared_group"
    # Single-member group: trivially one-session and time-axis-consistent.
    (SharedArtifactGroup & {"shared_artifact_group_name": group_name}).delete(
        safemode=False
    )
    SharedArtifactGroup.insert_group(group_name, [{"recording_id": rec_id}])

    expected = deterministic_id(
        "artifact_detection",
        {
            "source_kind": "shared_artifact_group",
            "artifact_detection_params_name": "default",
            "shared_artifact_group_name": group_name,
        },
    )
    select_key = {
        "shared_artifact_group_name": group_name,
        "artifact_detection_params_name": "default",
    }
    try:
        pk = ArtifactDetectionSelection.insert_selection(dict(select_key))
        assert pk == {"artifact_detection_id": expected}
        # Idempotent re-call returns the same id.
        assert (
            ArtifactDetectionSelection.insert_selection(dict(select_key)) == pk
        )
        # source_kind keeps it distinct from the recording-source id for the
        # same params + the member recording.
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
        (
            ArtifactDetectionSelection & {"artifact_detection_id": expected}
        ).delete(safemode=False)
        (
            SharedArtifactGroup & {"shared_artifact_group_name": group_name}
        ).delete(safemode=False)


# --------------------------------------------------------------------------
# Master-row identity immutability: in-place ``update1`` is rejected on every
# selection master AND on the ``CurationV2`` / ``SessionGroup`` factory masters
# (their columns are identity / provenance roots that dependents reference).
# Mirrors ``test_direct_master_insert_rejected_all_masters`` (the insert side)
# and ``test_sorter_parameters_update1_rejected_in_place`` (the param-Lookup
# side). The guard raises BEFORE any DB write, so a synthetic PK row never
# reaches MySQL -- no fixture rows needed for the reject path.
# --------------------------------------------------------------------------

_MASTER_UPDATE1_CASES = [
    ("spyglass.spikesorting.v2.recording", "RecordingSelection"),
    ("spyglass.spikesorting.v2.artifact", "ArtifactDetectionSelection"),
    ("spyglass.spikesorting.v2.sorting", "SortingSelection"),
    ("spyglass.spikesorting.v2.unit_matching", "UnitMatchSelection"),
    (
        "spyglass.spikesorting.v2.session_group",
        "ConcatenatedRecordingSelection",
    ),
    ("spyglass.spikesorting.v2.metric_curation", "CurationEvaluationSelection"),
    ("spyglass.spikesorting.v2.curation", "CurationV2"),
    ("spyglass.spikesorting.v2.session_group", "SessionGroup"),
]


def _synthetic_master_pk(cls):
    """Build a syntactically valid PK dict for a master (no DB row needed).

    UUID-keyed selection masters get a random ``uuid4`` PK; the composite-key
    factory masters (``CurationV2`` / ``SessionGroup``) get plausible literals.
    The guard rejects before any PK lookup, so the values need only fill the
    primary-key columns.
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
    ],
)
def test_factory_master_direct_insert_rejected(module, cls_name):
    """``CurationV2`` and ``SessionGroup`` are identity/provenance roots, not
    selection masters, but a direct ``insert1`` is blocked just the same (write
    them through ``insert_curation`` / ``create_group``). The guard fires before
    the FK checks, so a synthetic row never reaches MySQL. The factory paths
    themselves are covered by the ``single_session/test_curation_*`` and
    ``test_session_group_concat`` regression suites (which would break if the
    bypass were mis-wired)."""
    import importlib

    import datajoint as dj

    cls = getattr(importlib.import_module(module), cls_name)
    row = _synthetic_master_pk(cls)
    with pytest.raises(dj.errors.DataJointError, match="is not supported"):
        cls().insert1(row)


@pytest.mark.slow
@pytest.mark.integration
def test_master_update1_bypass_allows_mutation(fresh_recording_identity):
    """``allow_master_mutation=True`` forwards to the real ``update1`` so a
    deliberate maintenance edit of a row with no live references goes through --
    proving the escape hatch is wired, not just the reject path. Uses
    ``RecordingSelection`` (cheapest real-FK master)."""
    import datajoint as dj

    from spyglass.spikesorting.v2.recording import RecordingSelection

    pk = RecordingSelection.insert_selection(fresh_recording_identity)
    # Without the flag: rejected, row unchanged.
    with pytest.raises(dj.errors.DataJointError, match="is not supported"):
        RecordingSelection().update1({**pk, "team_name": "other_team"})
    # With the flag the guard forwards to super().update1; the team_name is a
    # real FK, so use a value that exists (the row's own team) to prove the
    # call reaches DataJoint rather than raising at the guard.
    current_team = (RecordingSelection & pk).fetch1("team_name")
    RecordingSelection().update1(
        {**pk, "team_name": current_team}, allow_master_mutation=True
    )
    assert (RecordingSelection & pk).fetch1("team_name") == current_team
