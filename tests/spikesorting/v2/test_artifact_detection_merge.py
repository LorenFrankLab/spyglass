"""``SortingSelection`` composition with the ``ArtifactDetectionOutput`` merge.

Covers atomic composition, delete lifecycle and ``get_merge_id`` integrity. No sorter or analyzer runs.
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
def ingested_recording(dj_conn, smoke_nwb):
    """Ingest the smoke fixture, a Recording, and a no-detect artifact detection.

    Yields ``rec_pk``, ``art_pk`` (a populated ``RecordingArtifactDetection``),
    and ``session``.
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
    """One insert_selection lands the recording-source part and the artifact
    registration together.

    There is one ``RecordingSource`` part, no ``ConcatenatedRecordingSource``
    part, one ``ArtifactDetectionSource`` part whose merge id resolves to the
    natural artifact id, and one ``ArtifactDetectionOutput`` registration.
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
        # The part stores the merge id, which resolves to the natural id.
        resolved_natural = (
            ArtifactDetectionOutput.resolve_artifact_detection_id(src[0])
        )
        assert str(resolved_natural) == str(art_pk["artifact_detection_id"])
        assert str(SortingSelection.resolve_artifact_detection(sort_pk)) == str(
            art_pk["artifact_detection_id"]
        )
        merge_id = ArtifactDetectionOutput.get_merge_id(
            {"artifact_detection_id": art_pk["artifact_detection_id"]}
        )
        assert len(ArtifactDetectionOutput & {"merge_id": merge_id}) == 1
    finally:
        (SortingSelection & sort_pk).super_delete(warn=False)


def test_artifact_output_delete_removes_source_part(ingested_recording):
    """Deleting the ArtifactDetectionOutput row leaves no dangling
    ``ArtifactDetectionSource`` part.

    The ``force_masters`` cascade removes the part and its ``SortingSelection``
    master. Uses a second detection so the shared fixture is untouched.
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

        (ArtifactDetectionOutput & {"merge_id": merge_id}).super_delete(
            warn=False, force_masters=True
        )

        # A part cannot outlive its master, so the SortingSelection goes too.
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
    """An artifact-free sort's ``sorting_id`` is the deterministic id of the
    recording-only payload with ``artifact_detection_id=None``."""
    from spyglass.spikesorting.v2._core.selection_identity import (
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
    """Populate a second ('default'-params) RecordingArtifactDetection."""
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
    """Two sorts on one recording and artifact detection share a single merge
    registration.

    The detection registers itself once at materialization, so there is no
    per-sort registration race.
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
        # One merge master backs the shared artifact...
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
    """An artifact-free insert_selection is safe under an ambient transaction.

    It touches no merge, so the master and recording source part land
    together on commit.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_pk = ingested_recording["rec_pk"]
    conn = dj.conn()
    sort_pk = None
    try:
        with conn.transaction:
            sort_pk = SortingSelection.insert_selection(
                {
                    "recording_id": rec_pk["recording_id"],
                    "sorter": "clusterless_thresholder",
                    "sorter_params_name": "default",
                }
            )
        # After commit: master and recording part present, no artifact part.
        assert SortingSelection & sort_pk
        assert SortingSelection.RecordingSource & sort_pk
        assert not (SortingSelection.ArtifactDetectionSource & sort_pk)
    finally:
        if sort_pk is not None:
            (SortingSelection & sort_pk).super_delete(warn=False)


def test_delete_detection_refused_when_referenced(ingested_recording):
    """Deleting an artifact result a sorting references is refused, leaving
    the detection and sort intact."""
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
        assert RecordingArtifactDetection & art
        assert SortingSelection & sort_pk
    finally:
        (SortingSelection & sort_pk).super_delete(warn=False)
        (RecordingArtifactDetection & art).cascade_delete(safemode=False)
        (RecordingArtifactSelection & art).super_delete(warn=False)


def test_delete_detection_unreferenced_unregisters_and_removes(
    ingested_recording,
):
    """Deleting an unreferenced artifact result removes it and its
    ArtifactDetectionOutput registration together."""
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput

    rec_pk = ingested_recording["rec_pk"]
    art = _second_detection(rec_pk)
    art_key = {"artifact_detection_id": art["artifact_detection_id"]}
    # Registered at materialization.
    merge_id = ArtifactDetectionOutput.get_merge_id(art_key)
    assert ArtifactDetectionOutput & {"merge_id": merge_id}
    try:
        (RecordingArtifactDetection & art).delete(safemode=False)
        # Detection and registration are gone; get_merge_id reports unregistered.
        assert not (RecordingArtifactDetection & art)
        assert not (ArtifactDetectionOutput & {"merge_id": merge_id})
        with pytest.raises(KeyError):
            ArtifactDetectionOutput.get_merge_id(art_key)
    finally:
        (RecordingArtifactSelection & art).super_delete(warn=False)


def test_cascade_delete_removes_referencing_sorts(ingested_recording):
    """cascade_delete removes the artifact result, its registration, and every
    sorting that referenced it."""
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


def test_insert_selection_lazily_registers_unregistered_detection(
    ingested_recording,
):
    """insert_selection re-registers a materialized but unregistered detection
    before linking it."""
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_pk = ingested_recording["rec_pk"]
    art = _second_detection(rec_pk)  # registered at populate
    art_key = {"artifact_detection_id": art["artifact_detection_id"]}
    merge_id = ArtifactDetectionOutput.get_merge_id(art_key)
    # Drop only the merge registration; the detection row survives.
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


def test_insert_detection_maps_only_missing_parent_to_key_error(
    dj_conn, monkeypatch
):
    """Only ``_merge_insert``'s missing-parent ``ValueError`` becomes the
    "not populated" ``KeyError``; any other ``ValueError`` propagates.
    """
    from spyglass.spikesorting.v2.artifact_output import (
        ArtifactDetectionOutput,
    )

    def merge_insert_raising(message):
        def _merge_insert(self, rows, **kwargs):
            raise ValueError(message)

        return _merge_insert

    key = {"artifact_detection_id": "not-a-detection"}
    monkeypatch.setattr(
        ArtifactDetectionOutput,
        "_merge_insert",
        merge_insert_raising(
            f"Non-existing entry in any of the parent tables - Entry: {key}"
        ),
    )
    with pytest.raises(KeyError, match="not a populated"):
        ArtifactDetectionOutput.insert_detection(key)

    monkeypatch.setattr(
        ArtifactDetectionOutput,
        "_merge_insert",
        merge_insert_raising("Ambiguous entry. Data has mult rows"),
    )
    with pytest.raises(ValueError, match="Ambiguous entry"):
        ArtifactDetectionOutput.insert_detection(key)


def test_get_merge_id_distinguishes_unregistered_from_corrupt(dj_conn):
    """``get_merge_id`` raises ``KeyError`` for an unregistered id but
    ``SchemaBypassError`` for a corrupt >1-row registration.

    Callers treat ``KeyError`` as "no existing selection", so a corrupt
    registration must not raise it.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.artifact_output import (
        ArtifactDetectionOutput,
    )
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    with pytest.raises(KeyError):
        ArtifactDetectionOutput.get_merge_id(
            {"artifact_detection_id": uuid.uuid4()}
        )

    # Corrupt: two RecordingSource rows for one artifact_detection_id (only
    # reachable via a raw insert bypassing insert_detection).
    art_id = uuid.uuid4()
    planted_merge_ids = []
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        for _ in range(2):
            mid = uuid.uuid4()
            planted_merge_ids.append(mid)
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
            for mid in planted_merge_ids:
                (ArtifactDetectionOutput & {"merge_id": mid}).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")


def test_get_merge_id_cross_part_duplicate_raises(dj_conn):
    """``get_merge_id`` raises ``SchemaBypassError`` when one id is registered
    in both ``RecordingSource`` and ``SharedGroupSource``."""
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
        # Same id in different source parts, each under its own master (raw
        # insert only).
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


def test_insert_selection_links_artifact_in_ambient_transaction(
    ingested_recording,
):
    """An artifact-bound insert_selection joins a caller-owned transaction and
    rolls back with it."""
    import datajoint as dj

    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_pk = ingested_recording["rec_pk"]
    art = _second_detection(rec_pk)
    sel = {
        "recording_id": rec_pk["recording_id"],
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "default",
        "artifact_detection_id": art["artifact_detection_id"],
    }
    conn = dj.conn()
    n_before = len(SortingSelection())
    try:
        conn.start_transaction()
        try:
            sort_pk = SortingSelection.insert_selection(sel)
            assert SortingSelection & sort_pk
        finally:
            conn.cancel_transaction()
        assert len(SortingSelection()) == n_before
    finally:
        (RecordingArtifactDetection & art).cascade_delete(safemode=False)
        (RecordingArtifactSelection & art).super_delete(warn=False)


def test_artifact_source_fk_rejects_dangling_detection(ingested_recording):
    """The ``ArtifactDetectionSource -> ArtifactDetectionOutput`` foreign key
    rejects a row referencing a non-existent merge."""
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import SortingSelection

    rec_pk = ingested_recording["rec_pk"]
    # An artifact-free master to hang a bogus artifact part on.
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        }
    )
    try:
        with pytest.raises(dj.errors.IntegrityError):
            SortingSelection.ArtifactDetectionSource.insert1(
                {
                    "sorting_id": sort_pk["sorting_id"],
                    "artifact_detection_merge_id": uuid.uuid4(),
                }
            )
    finally:
        (SortingSelection & sort_pk).super_delete(warn=False)
