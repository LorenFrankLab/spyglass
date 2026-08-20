"""``SortingSelection`` composition with the ``ArtifactDetectionOutput`` merge.

Exercises the seam where a ``SortingSelection`` (recording-source XOR part) links
an artifact detection through the ``ArtifactDetectionOutput`` merge:

* ``insert_selection`` composition -- the recording-source part and the artifact
  registration land together in one transaction;
* lifecycle -- deleting a detection is refused while a sort references it, and a
  coordinated / cascade delete leaves no dangling ``ArtifactDetectionSource``
  part;
* concurrency -- the delete-vs-select advisory lock is fail-closed;
* merge integrity -- ``get_merge_id`` fails closed on a corrupt registration;
* the artifact-free ``sorting_id`` fold is unchanged.

Fast: no sorter and no analyzer are run -- only selections + registrations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SMOKE = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_smoke.nwb"
)
_SETUP = {
    "sort_group_id": 0,
    "interval_list_name": "raw data valid times",
    "preprocessing_params_name": "default",
    "team_name": "artifact_merge_team",
    "sorter": "mountainsort5",
    "sorter_params_name": "franklab_30khz_ms5_2026_06",
}


@pytest.fixture(scope="module")
def ingested_recording(dj_conn):
    """Ingest the smoke fixture + a Recording + a no-detect artifact detection.

    Module-scoped so the ingest runs once; individual tests build their own
    cheap ``SortingSelection`` rows on top and clean them up. Yields ``rec_pk``,
    ``art_pk`` (a populated ``RecordingArtifactDetection``), and ``session``.
    """
    if not _SMOKE.exists():
        pytest.skip(f"smoke MEArec fixture not present ({_SMOKE})")

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        copy_and_insert_nwb,
    )

    nwb = copy_and_insert_nwb(_SMOKE, dest_name="artifact_merge.nwb")
    session = {"nwb_file_name": nwb}
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": _SETUP["team_name"], "team_description": "integration"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb,
            "sort_group_id": _SETUP["sort_group_id"],
            "interval_list_name": _SETUP["interval_list_name"],
            "preprocessing_params_name": _SETUP["preprocessing_params_name"],
            "team_name": _SETUP["team_name"],
        }
    )
    if not (Recording & rec_pk):
        Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (RecordingArtifactDetection & art_pk):
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)

    yield {"rec_pk": rec_pk, "art_pk": art_pk, "session": session}
    _clean_session_v2(session)


def test_insert_selection_composes_xor_and_artifact_atomically(
    ingested_recording,
):
    """One insert_selection lands the recording-source XOR part
    AND the artifact registration together.

    After a single artifact-backed ``insert_selection`` there is exactly one
    ``RecordingSource`` part (the recording XOR), no ``ConcatenatedRecordingSource``
    part, one ``ArtifactDetectionSource`` part carrying an
    ``artifact_detection_merge_id`` that resolves to the natural artifact id, and
    the detection is registered once in ``ArtifactDetectionOutput``.
    """
    from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_pk = ingested_recording["rec_pk"]
    art_pk = ingested_recording["art_pk"]
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": _SETUP["sorter"],
            "sorter_params_name": _SETUP["sorter_params_name"],
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    try:
        assert len(SortingSelection.RecordingSource & sort_pk) == 1
        assert len(SortingSelection.ConcatenatedRecordingSource & sort_pk) == 0

        src = (SortingSelection.ArtifactDetectionSource & sort_pk).fetch(
            "artifact_detection_merge_id"
        )
        assert len(src) == 1
        # The part stores the MERGE id, which resolves back to the natural id.
        resolved_natural = (
            ArtifactDetectionOutput.resolve_artifact_detection_id(src[0])
        )
        assert str(resolved_natural) == str(art_pk["artifact_detection_id"])
        assert str(SortingSelection.resolve_artifact_detection(sort_pk)) == str(
            art_pk["artifact_detection_id"]
        )
        # The detection is registered exactly once in the merge.
        merge_id = ArtifactDetectionOutput.get_merge_id(
            {"artifact_detection_id": art_pk["artifact_detection_id"]}
        )
        assert len(ArtifactDetectionOutput & {"merge_id": merge_id}) == 1
    finally:
        (SortingSelection & sort_pk).super_delete(warn=False)


def test_artifact_output_delete_removes_source_part(ingested_recording):
    """Deleting the ArtifactDetectionOutput row leaves no
    dangling ``ArtifactDetectionSource`` part.

    Builds a SECOND artifact detection (distinct params) + an artifact-backed
    sort so the shared no-detect fixture is untouched, then deletes the merge row
    the sort references. The ``ArtifactDetectionSource`` part FK's the merge, so
    a ``force_masters`` cascade must remove that part -- and, because DataJoint
    cannot strand a part without its master, the dependent ``SortingSelection``
    with it. The guarantee under test is that NOTHING is left dangling.
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_pk = ingested_recording["rec_pk"]
    ArtifactDetectionParameters.insert_default()
    art2 = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "default",
        }
    )
    if not (RecordingArtifactDetection & art2):
        RecordingArtifactDetection.populate(art2, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": _SETUP["sorter"],
            "sorter_params_name": _SETUP["sorter_params_name"],
            "artifact_detection_id": art2["artifact_detection_id"],
        }
    )
    try:
        assert SortingSelection.ArtifactDetectionSource & sort_pk
        merge_id = ArtifactDetectionOutput.get_merge_id(art2)

        # Delete the merge row the sort's artifact part references.
        (ArtifactDetectionOutput & {"merge_id": merge_id}).super_delete(
            warn=False, force_masters=True
        )

        # The force_masters cascade removes the ArtifactDetectionSource part and,
        # because a part cannot be stranded without its master, the dependent
        # SortingSelection (master + its RecordingSource part) too. Nothing is
        # left dangling.
        assert not (
            SortingSelection.ArtifactDetectionSource & sort_pk
        ), "ArtifactDetectionSource part dangles after ArtifactDetectionOutput delete"
        assert not (
            SortingSelection & sort_pk
        ), "SortingSelection master dangles"
        assert not (
            SortingSelection.RecordingSource & sort_pk
        ), "RecordingSource part dangles"
    finally:
        (SortingSelection & sort_pk).super_delete(warn=False)
        (RecordingArtifactDetection & art2).super_delete(warn=False)
        (RecordingArtifactSelection & art2).super_delete(warn=False)


def test_artifact_free_sorting_id_matches_recording_only_payload(
    ingested_recording,
):
    """Artifact-free: an artifact-free sort's ``sorting_id`` is the
    deterministic id of the recording-only payload with ``artifact_detection_id=None``.

    ``_selection_identity`` is byte-stable, so this pins that the
    artifact-free fold is unchanged (no ArtifactDetectionSource row participates,
    and the id is not aliased by the merge seam).
    """
    from spyglass.spikesorting.v2._selection_identity import (
        deterministic_id,
        sorting_identity_payload,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_pk = ingested_recording["rec_pk"]
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": _SETUP["sorter"],
            "sorter_params_name": _SETUP["sorter_params_name"],
        }
    )
    try:
        assert SortingSelection.resolve_artifact_detection(sort_pk) is None
        assert len(SortingSelection.ArtifactDetectionSource & sort_pk) == 0
        expected = deterministic_id(
            "sorting",
            sorting_identity_payload(
                recording_id=rec_pk["recording_id"],
                sorter=_SETUP["sorter"],
                sorter_params_name=_SETUP["sorter_params_name"],
                artifact_detection_id=None,
            ),
        )
        assert str(sort_pk["sorting_id"]) == str(expected)
    finally:
        (SortingSelection & sort_pk).super_delete(warn=False)


def _second_detection(rec_pk):
    """Populate a SECOND ('default'-params) RecordingArtifactDetection."""
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )

    ArtifactDetectionParameters.insert_default()
    art = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "default",
        }
    )
    if not (RecordingArtifactDetection & art):
        RecordingArtifactDetection.populate(art, reserve_jobs=False)
    return art


def test_two_sorts_one_artifact_share_single_registration(ingested_recording):
    """two DIFFERENT sorts on one
    recording + one artifact detection share a SINGLE merge registration.

    Producer-owned registration means the detection registers itself ONCE at
    materialization; the two sorts (different sorters) each just resolve that one
    merge id. There is no per-sort registration and hence no merge-master race:
    exactly one ArtifactDetectionOutput master backs both sorts.
    """
    from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_pk = ingested_recording["rec_pk"]
    art_pk = ingested_recording["art_pk"]
    art_id = art_pk["artifact_detection_id"]
    merge_id = ArtifactDetectionOutput.get_merge_id(
        {"artifact_detection_id": art_id}
    )
    sorts = []
    try:
        for sorter, params in (
            ("mountainsort5", "franklab_30khz_ms5_2026_06"),
            ("clusterless_thresholder", "default"),
        ):
            sorts.append(
                SortingSelection.insert_selection(
                    {
                        "recording_id": rec_pk["recording_id"],
                        "sorter": sorter,
                        "sorter_params_name": params,
                        "artifact_detection_id": art_id,
                    }
                )
            )
        assert sorts[0]["sorting_id"] != sorts[1]["sorting_id"]
        # Exactly ONE merge master backs the shared artifact...
        assert len(ArtifactDetectionOutput & {"merge_id": merge_id}) == 1
        # ...and both sorts' ArtifactDetectionSource parts point at it.
        for sort_pk in sorts:
            got = (SortingSelection.ArtifactDetectionSource & sort_pk).fetch1(
                "artifact_detection_merge_id"
            )
            assert str(got) == str(merge_id)
    finally:
        for sort_pk in sorts:
            (SortingSelection & sort_pk).super_delete(warn=False)


def test_insert_selection_safe_inside_outer_transaction(ingested_recording):
    """insert_selection is safe under
    an ambient transaction.

    The producer-owned redesign removes the merge insert from the sorting
    transaction, so calling insert_selection inside an outer transaction opens no
    nested transaction and strands no partial rows -- the master, recording
    source part, and artifact source part all land together.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_pk = ingested_recording["rec_pk"]
    art_pk = ingested_recording["art_pk"]
    conn = dj.conn()
    sort_pk = None
    try:
        with conn.transaction:
            sort_pk = SortingSelection.insert_selection(
                {
                    "recording_id": rec_pk["recording_id"],
                    "sorter": "clusterless_thresholder",
                    "sorter_params_name": "default",
                    "artifact_detection_id": art_pk["artifact_detection_id"],
                }
            )
        # After the outer transaction commits, all three rows are present.
        assert SortingSelection & sort_pk
        assert SortingSelection.RecordingSource & sort_pk
        assert SortingSelection.ArtifactDetectionSource & sort_pk
    finally:
        if sort_pk is not None:
            (SortingSelection & sort_pk).super_delete(warn=False)


def test_delete_detection_refused_when_referenced(ingested_recording):
    """Deleting an artifact result a
    sorting references is REFUSED, leaving the detection and sort intact."""
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_pk = ingested_recording["rec_pk"]
    art = _second_detection(rec_pk)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "artifact_detection_id": art["artifact_detection_id"],
        }
    )
    try:
        with pytest.raises(
            ValueError, match="referenced by a SortingSelection"
        ):
            (RecordingArtifactDetection & art).delete(safemode=False)
        # Nothing was removed.
        assert RecordingArtifactDetection & art
        assert SortingSelection & sort_pk
    finally:
        (SortingSelection & sort_pk).super_delete(warn=False)
        (RecordingArtifactDetection & art).cascade_delete(safemode=False)
        (RecordingArtifactSelection & art).super_delete(warn=False)


def test_delete_detection_unreferenced_unregisters_and_removes(
    ingested_recording,
):
    """Deleting an UNREFERENCED artifact result removes it
    and its ArtifactDetectionOutput registration together."""
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput

    rec_pk = ingested_recording["rec_pk"]
    art = _second_detection(rec_pk)
    art_key = {"artifact_detection_id": art["artifact_detection_id"]}
    # Registered at materialization (producer-owned).
    merge_id = ArtifactDetectionOutput.get_merge_id(art_key)
    assert ArtifactDetectionOutput & {"merge_id": merge_id}
    try:
        (RecordingArtifactDetection & art).delete(safemode=False)
        # Both the detection and its merge registration are gone; get_merge_id
        # now reports it as unregistered.
        assert not (RecordingArtifactDetection & art)
        assert not (ArtifactDetectionOutput & {"merge_id": merge_id})
        with pytest.raises(KeyError):
            ArtifactDetectionOutput.get_merge_id(art_key)
    finally:
        (RecordingArtifactSelection & art).super_delete(warn=False)


def test_cascade_delete_removes_referencing_sorts(ingested_recording):
    """cascade_delete removes the artifact result, its
    registration, AND every sorting that referenced it."""
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_pk = ingested_recording["rec_pk"]
    art = _second_detection(rec_pk)
    merge_id = ArtifactDetectionOutput.get_merge_id(
        {"artifact_detection_id": art["artifact_detection_id"]}
    )
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "artifact_detection_id": art["artifact_detection_id"],
        }
    )
    try:
        (RecordingArtifactDetection & art).cascade_delete(safemode=False)
        assert not (RecordingArtifactDetection & art)
        assert not (ArtifactDetectionOutput & {"merge_id": merge_id})
        assert not (SortingSelection & sort_pk)
    finally:
        (SortingSelection & sort_pk).super_delete(warn=False)
        (RecordingArtifactSelection & art).super_delete(warn=False)


def test_merge_source_part_integrity_is_clean(ingested_recording):
    """A well-formed tree has no orphan/ambiguous merge
    masters (every ArtifactDetectionOutput master has exactly one source part).
    """
    from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput

    # The fixture's populated detection is registered; the audit must be clean.
    assert ArtifactDetectionOutput.audit_source_part_integrity() == []


def test_merge_integrity_flags_orphan_master(ingested_recording):
    """audit_source_part_integrity flags
    a raw-inserted merge master that has NO source part.

    Producer-owned registration + coordinated deletion keep the merge
    well-formed, so a bad master is only reachable by a raw insert; plant one
    with FK-checks-off and confirm the audit reports it (count 0,
    discriminator mismatch)."""
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput

    orphan_id = uuid.uuid4()
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        # dj.Table.insert bypasses the _Merge insert override to plant a bare
        # master with no source part.
        dj.Table.insert(
            ArtifactDetectionOutput(),
            [{"merge_id": orphan_id, "source": "RecordingSource"}],
        )
        flagged = ArtifactDetectionOutput.audit_source_part_integrity()
        entry = next((f for f in flagged if f["merge_id"] == orphan_id), None)
        assert entry is not None, "orphan merge master not flagged by the audit"
        assert entry["source_part_count"] == 0
        assert entry["discriminator_ok"] is False
    finally:
        (ArtifactDetectionOutput & {"merge_id": orphan_id}).super_delete(
            warn=False, force_masters=True
        )
        conn.query("SET FOREIGN_KEY_CHECKS=1")


def test_insert_selection_lazily_registers_unregistered_detection(
    ingested_recording,
):
    """insert_selection re-registers a
    materialized-but-unregistered detection before linking it.

    Producer-owned registration means a populated detection is normally already
    in the merge, so this drops ONLY the ArtifactDetectionOutput registration
    (keeping the RecordingArtifactDetection row) to reach insert_selection's lazy
    register-if-absent fallback, then asserts the sort links through it."""
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_pk = ingested_recording["rec_pk"]
    art = _second_detection(rec_pk)  # populated -> producer-owned registered
    art_key = {"artifact_detection_id": art["artifact_detection_id"]}
    merge_id = ArtifactDetectionOutput.get_merge_id(art_key)
    # Drop ONLY the merge registration (force_masters removes its source part);
    # the RecordingArtifactDetection row survives -> get_merge_id now KeyErrors.
    (ArtifactDetectionOutput & {"merge_id": merge_id}).super_delete(
        warn=False, force_masters=True
    )
    sort_pk = None
    try:
        with pytest.raises(KeyError):
            ArtifactDetectionOutput.get_merge_id(art_key)
        assert RecordingArtifactDetection & art, "detection row should survive"

        sort_pk = SortingSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "sorter": "clusterless_thresholder",
                "sorter_params_name": "default",
                "artifact_detection_id": art["artifact_detection_id"],
            }
        )
        # The lazy fallback re-registered it, and the sort links through it.
        assert ArtifactDetectionOutput.get_merge_id(art_key)
        assert SortingSelection.ArtifactDetectionSource & sort_pk
        assert str(SortingSelection.resolve_artifact_detection(sort_pk)) == str(
            art["artifact_detection_id"]
        )
    finally:
        if sort_pk is not None:
            (SortingSelection & sort_pk).super_delete(warn=False)
        (RecordingArtifactDetection & art).super_delete(
            warn=False, force_masters=True
        )
        (RecordingArtifactSelection & art).super_delete(warn=False)


def test_delete_and_insert_lock_the_same_detection_key(
    ingested_recording, monkeypatch
):
    """Serialization wiring: both insert_selection and
    the detection delete take the advisory lock keyed on the SAME
    (ArtifactDetectionOutput, artifact_detection_id).

    A true two-connection race test is a flaky anti-pattern; instead this pins
    that the two sides contend on ONE lock -- the mechanism that closes the
    delete-vs-select race -- so a refactor that drops the lock on either side is
    caught deterministically.
    """
    import contextlib

    import spyglass.spikesorting.v2._db_locking as db_locking
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recorded: list = []
    real_lock = db_locking.required_advisory_lock

    @contextlib.contextmanager
    def spy_lock(table, key, **kwargs):
        recorded.append((table.full_table_name, dict(key)))
        with real_lock(table, key, **kwargs):
            yield

    # Both insert_selection and delete import required_advisory_lock from
    # _db_locking at call time, so patching it there covers both.
    monkeypatch.setattr(db_locking, "required_advisory_lock", spy_lock)

    rec_pk = ingested_recording["rec_pk"]
    art = _second_detection(rec_pk)
    art_id = art["artifact_detection_id"]
    merge_tbl = ArtifactDetectionOutput().full_table_name
    sort_pk = None
    try:
        sort_pk = SortingSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "sorter": "clusterless_thresholder",
                "sorter_params_name": "default",
                "artifact_detection_id": art_id,
            }
        )
        insert_locks = [
            (t, k)
            for t, k in recorded
            if t == merge_tbl
            and str(k.get("artifact_detection_id")) == str(art_id)
        ]
        assert (
            insert_locks
        ), "insert_selection did not lock the artifact detection"

        recorded.clear()
        (RecordingArtifactDetection & art).cascade_delete(safemode=False)
        delete_locks = [
            (t, k)
            for t, k in recorded
            if t == merge_tbl
            and str(k.get("artifact_detection_id")) == str(art_id)
        ]
        assert delete_locks, "delete did not lock the artifact detection"
    finally:
        if sort_pk is not None and (SortingSelection & sort_pk):
            (SortingSelection & sort_pk).super_delete(warn=False)
        if RecordingArtifactDetection & art:
            (RecordingArtifactDetection & art).super_delete(
                warn=False, force_masters=True
            )
        (RecordingArtifactSelection & art).super_delete(warn=False)


def test_required_lock_failure_aborts_insert_and_delete(
    ingested_recording, monkeypatch
):
    """Fail-closed: when the advisory lock CANNOT be acquired (it raises
    ``AdvisoryLockError``), neither ``insert_selection`` nor the detection delete
    mutates the database.

    The wiring test proves both paths CALL the lock; this proves a FAILED
    acquisition STOPS the operation -- the property that makes "fail-closed" real
    rather than a label. Simulated by patching the lock to raise, the same way a
    ``GET_LOCK`` timeout/error surfaces from ``required_advisory_lock``.
    """
    import spyglass.spikesorting.v2._db_locking as db_locking
    from spyglass.spikesorting.v2._db_locking import AdvisoryLockError
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput
    from spyglass.spikesorting.v2.sorting import SortingSelection

    def boom(table, key, **kwargs):
        raise AdvisoryLockError("simulated acquisition failure")

    rec_pk = ingested_recording["rec_pk"]
    art = _second_detection(rec_pk)
    art_id = art["artifact_detection_id"]
    merge_id = ArtifactDetectionOutput.get_merge_id(
        {"artifact_detection_id": art_id}
    )
    sel = {
        "recording_id": rec_pk["recording_id"],
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "default",
        "artifact_detection_id": art_id,
    }
    n_sel_before = len(SortingSelection())
    try:
        monkeypatch.setattr(db_locking, "required_advisory_lock", boom)

        # insert_selection aborts before its transaction -> writes NOTHING.
        with pytest.raises(AdvisoryLockError):
            SortingSelection.insert_selection(sel)
        assert (
            len(SortingSelection()) == n_sel_before
        ), "insert_selection created a row despite a failed lock"
        assert not (
            SortingSelection.ArtifactDetectionSource
            & {"artifact_detection_merge_id": merge_id}
        ), "insert_selection linked a sort despite a failed lock"

        # delete of the (still unreferenced) detection aborts -> removes NOTHING.
        with pytest.raises(AdvisoryLockError):
            (RecordingArtifactDetection & art).delete(safemode=False)
        assert RecordingArtifactDetection & art
        assert (
            ArtifactDetectionOutput.get_merge_id(
                {"artifact_detection_id": art_id}
            )
            == merge_id
        )
    finally:
        monkeypatch.undo()  # restore the real lock before teardown deletes
        (RecordingArtifactDetection & art).cascade_delete(safemode=False)
        (RecordingArtifactSelection & art).super_delete(warn=False)


def test_held_lock_blocks_insert_and_delete(ingested_recording, monkeypatch):
    """Deterministic (no sleeps) proof the exclusion is REAL: while a SEPARATE DB
    session holds the detection's named lock, ``insert_selection`` and the
    detection delete both time out and raise ``AdvisoryLockError`` -- a waiting
    operation cannot slip past a concurrently-held lock. Once the lock is
    released, both paths acquire it: the blocked ``insert_selection`` commits
    (creating a referrer), and the delete path then behaves correctly under the
    now-free lock -- an ordinary delete is refused (referrer present) while
    ``cascade_delete`` acquires and removes the detection with its sort.

    MySQL named locks are reentrant WITHIN a session, so a same-session check
    could pass while the lock is a no-op; only an independent second connection
    that must block proves the serialization actually excludes.
    """
    import datajoint as dj

    import spyglass.spikesorting.v2._db_locking as db_locking
    from spyglass.spikesorting.v2._db_locking import (
        AdvisoryLockError,
        _lock_name,
    )
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput
    from spyglass.spikesorting.v2.sorting import SortingSelection

    # Shorten the lifecycle timeout so a blocked acquire fails in ~1s, not 120.
    # Read at call time inside required_advisory_lock, so the patch takes effect.
    monkeypatch.setattr(db_locking, "LIFECYCLE_LOCK_TIMEOUT_S", 1)

    rec_pk = ingested_recording["rec_pk"]
    art = _second_detection(rec_pk)
    art_id = art["artifact_detection_id"]
    lock_name = _lock_name(
        ArtifactDetectionOutput, {"artifact_detection_id": art_id}
    )
    conn_b = dj.Connection(
        dj.config["database.host"],
        dj.config["database.user"],
        dj.config["database.password"],
        port=dj.config["database.port"],
    )
    sel = {
        "recording_id": rec_pk["recording_id"],
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "default",
        "artifact_detection_id": art_id,
    }
    sort_pk = None
    try:
        held = conn_b.query(
            "SELECT GET_LOCK(%s, %s)", args=(lock_name, 10)
        ).fetchone()
        assert held and held[0] == 1, "second session could not take the lock"

        # Both lifecycle ops block on the held lock and fail closed.
        with pytest.raises(AdvisoryLockError):
            SortingSelection.insert_selection(sel)
        with pytest.raises(AdvisoryLockError):
            (RecordingArtifactDetection & art).delete(safemode=False)
        # The detection survived (its delete never ran past the lock).
        assert RecordingArtifactDetection & art

        # Release the lock -> the previously-blocked operations acquire it.
        # The blocked insert_selection commits, creating a referrer.
        conn_b.query("SELECT RELEASE_LOCK(%s)", args=(lock_name,))
        sort_pk = SortingSelection.insert_selection(sel)
        assert SortingSelection & sort_pk
        # The delete path acquires the now-free lock too: an ordinary delete
        # sees the fresh referrer and is REFUSED, while cascade_delete acquires
        # and removes the detection together with its sort.
        with pytest.raises(
            ValueError, match="referenced by a SortingSelection"
        ):
            (RecordingArtifactDetection & art).delete(safemode=False)
        (RecordingArtifactDetection & art).cascade_delete(safemode=False)
        assert not (RecordingArtifactDetection & art)
        assert not (SortingSelection & sort_pk)
    finally:
        conn_b.close()  # frees the held named lock if an assert left it held
        if sort_pk is not None and (SortingSelection & sort_pk):
            (SortingSelection & sort_pk).super_delete(warn=False)
        if RecordingArtifactDetection & art:
            (RecordingArtifactDetection & art).cascade_delete(safemode=False)
        (RecordingArtifactSelection & art).super_delete(warn=False)


def test_get_merge_id_distinguishes_unregistered_from_corrupt(dj_conn):
    """``get_merge_id`` raises ``KeyError`` for an unregistered id but
    ``SchemaBypassError`` for a corrupt >1-row registration.

    The distinction is load-bearing: callers catch ``KeyError`` to mean "no
    existing selection", so collapsing the corrupt case into ``KeyError`` would
    silently swallow a duplicate registration.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.artifact_output import (
        ArtifactDetectionOutput,
    )
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    # Unregistered id -> KeyError.
    with pytest.raises(KeyError):
        ArtifactDetectionOutput.get_merge_id(
            {"artifact_detection_id": uuid.uuid4()}
        )

    # Corrupt: two RecordingSource rows for one artifact_detection_id (only
    # reachable via a raw insert bypassing insert_detection).
    art_id = uuid.uuid4()
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        for _ in range(2):
            mid = uuid.uuid4()
            dj.Table.insert(
                ArtifactDetectionOutput(),
                [{"merge_id": mid, "source": "RecordingSource"}],
            )
            dj.Table.insert(
                ArtifactDetectionOutput.RecordingSource(),
                [{"merge_id": mid, "artifact_detection_id": art_id}],
            )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")
    try:
        with pytest.raises(SchemaBypassError, match="corrupt duplicate"):
            ArtifactDetectionOutput.get_merge_id(
                {"artifact_detection_id": art_id}
            )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (
                ArtifactDetectionOutput.RecordingSource
                & {"artifact_detection_id": art_id}
            ).delete_quick()
            for mid in (
                ArtifactDetectionOutput & {"source": "RecordingSource"}
            ).fetch("merge_id"):
                if not (
                    ArtifactDetectionOutput.RecordingSource & {"merge_id": mid}
                ):
                    (ArtifactDetectionOutput & {"merge_id": mid}).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")


def test_get_merge_id_cross_part_duplicate_raises(dj_conn):
    """``get_merge_id`` raises ``SchemaBypassError`` when one id is registered in
    BOTH source parts -- the cross-part case, distinct from a same-part double.

    Resolution trusts that a detection id lives in exactly one source part. A raw
    double-insert that puts the same id in ``RecordingSource`` AND
    ``SharedGroupSource`` is corruption; scanning only the first part would
    silently resolve to whichever is scanned first, so ``get_merge_id`` must scan
    both and fail closed. This is the exact condition the same-part corruption
    test does NOT construct.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.artifact_output import (
        ArtifactDetectionOutput,
    )
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    art_id = uuid.uuid4()
    rec_mid = uuid.uuid4()
    shared_mid = uuid.uuid4()
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        # Same artifact_detection_id registered in DIFFERENT source parts, each
        # under its own merge master (only reachable via a raw insert).
        dj.Table.insert(
            ArtifactDetectionOutput(),
            [
                {"merge_id": rec_mid, "source": "RecordingSource"},
                {"merge_id": shared_mid, "source": "SharedGroupSource"},
            ],
        )
        dj.Table.insert(
            ArtifactDetectionOutput.RecordingSource(),
            [{"merge_id": rec_mid, "artifact_detection_id": art_id}],
        )
        dj.Table.insert(
            ArtifactDetectionOutput.SharedGroupSource(),
            [{"merge_id": shared_mid, "artifact_detection_id": art_id}],
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")
    try:
        with pytest.raises(SchemaBypassError, match="cross-part"):
            ArtifactDetectionOutput.get_merge_id(
                {"artifact_detection_id": art_id}
            )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            for part in (
                ArtifactDetectionOutput.RecordingSource,
                ArtifactDetectionOutput.SharedGroupSource,
            ):
                (part & {"artifact_detection_id": art_id}).delete_quick()
            for mid in (rec_mid, shared_mid):
                (ArtifactDetectionOutput & {"merge_id": mid}).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
