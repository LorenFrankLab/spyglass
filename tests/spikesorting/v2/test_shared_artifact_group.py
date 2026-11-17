"""Artifact-detection invariants: shared-group guards and selection branches.

The create_group checks fire before any recording load, so preconditions are
planted with FK checks off instead of a real populate.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import _synthetic_artifact_recording


def _plant_fake_recording(recording_id, nwb_file_name, sampling_frequency):
    """Insert a minimal ``Recording`` and ``RecordingSelection`` pair with FK
    checks off.

    Sets only the ``nwb_file_name`` and ``sampling_frequency`` the
    create_group guards read. Returns the ``recording_id``.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection

    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        RecordingSelection.insert1(
            {
                "recording_id": recording_id,
                "nwb_file_name": nwb_file_name,
                "sort_group_id": 0,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "v2_a25_team",
                "recording_input_hash": "0" * 64,
            },
            allow_direct_insert=True,
        )
        Recording.insert1(
            {
                "recording_id": recording_id,
                "analysis_file_name": "a25_fake.nwb",
                "electrical_series_path": "/fake/es",
                "object_id": "a25-fake-object-id",
                "n_channels": 4,
                "sampling_frequency": sampling_frequency,
                "duration_s": 60.0,
                "content_hash": "0" * 64,
            },
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")
    return recording_id


def _drop_fake_recording(recording_id):
    """Tear down a ``_plant_fake_recording`` pair (parts-first)."""
    import datajoint as dj

    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection

    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        (Recording & {"recording_id": recording_id}).delete_quick()
        (RecordingSelection & {"recording_id": recording_id}).delete_quick()
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")


class _FakeRecording:
    """Stand-in for an explicit-time_vector SI recording.

    Implements the SI calls ``create_group`` uses (``get_time_info``,
    ``sample_index_to_time``, ``get_num_samples``) so the real fingerprint
    path runs.
    """

    def __init__(self, times, fs=30000.0):
        self._times = np.asarray(times, dtype=np.float64)
        self._fs = float(fs)

    def get_num_samples(self, segment_index=None):
        return len(self._times)

    def get_sampling_frequency(self):
        return self._fs

    def get_dtype(self):
        return "float32"

    def get_num_segments(self):
        return 1

    def get_times(self, segment_index=None):
        return self._times

    def get_time_info(self, segment_index=None):
        return {
            "sampling_frequency": self._fs,
            "t_start": float(self._times[0]),
            "time_vector": self._times,
        }

    def sample_index_to_time(self, sample_ind, segment_index=None):
        return self._times[sample_ind]


@pytest.mark.usefixtures("dj_conn")
def test_shared_artifact_group_rejects_cross_session_members():
    """Members from two sessions raise, and no master row is left.

    IntervalList rows are keyed by session, so a mixed-session group has
    undefined artifact-removed valid times.
    """
    import uuid

    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup

    rid_a = _plant_fake_recording(uuid.uuid4(), "session_a_.nwb", 30000.0)
    rid_b = _plant_fake_recording(uuid.uuid4(), "session_b_.nwb", 30000.0)
    group_name = "v2_a25_cross_session"
    try:
        with pytest.raises(ValueError, match="members span 2 sessions"):
            SharedArtifactGroup.create_group(
                "session_a_.nwb",
                group_name,
                [{"recording_id": rid_a}, {"recording_id": rid_b}],
            )
        assert (
            len(
                SharedArtifactGroup & {"shared_artifact_group_name": group_name}
            )
            == 0
        ), "master row not rolled back after cross-session raise"
    finally:
        _drop_fake_recording(rid_a)
        _drop_fake_recording(rid_b)


@pytest.mark.usefixtures("dj_conn")
def test_shared_artifact_group_rejects_frequency_mismatch():
    """Members with differing sampling frequencies raise.

    ``si.aggregate_channels`` requires identical fs. Both members share one
    session so only the frequency guard fires.
    """
    import uuid

    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup

    rid_a = _plant_fake_recording(uuid.uuid4(), "session_freq_.nwb", 30000.0)
    rid_b = _plant_fake_recording(uuid.uuid4(), "session_freq_.nwb", 25000.0)
    try:
        with pytest.raises(ValueError, match="differing sampling frequencies"):
            SharedArtifactGroup.create_group(
                "session_freq_.nwb",
                "v2_a25_freq_mismatch",
                [{"recording_id": rid_a}, {"recording_id": rid_b}],
            )
    finally:
        _drop_fake_recording(rid_a)
        _drop_fake_recording(rid_b)


@pytest.mark.usefixtures("dj_conn")
def test_shared_artifact_group_rejects_timestamp_mismatch(monkeypatch):
    """Equal-length same-session members still need equal timestamps.

    Only the wall-clock vectors differ; ``si.aggregate_channels`` would
    otherwise stack misaligned frames.
    """
    import uuid

    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup
    from spyglass.spikesorting.v2.recording import Recording

    rid_a = _plant_fake_recording(uuid.uuid4(), "session_times_.nwb", 30000.0)
    rid_b = _plant_fake_recording(uuid.uuid4(), "session_times_.nwb", 30000.0)

    base_times = np.arange(8, dtype=np.float64) / 30000.0

    def _fake_get_recording(self, key):
        if str(key["recording_id"]) == str(rid_a):
            return _FakeRecording(base_times)
        return _FakeRecording(base_times + 10.0)

    monkeypatch.setattr(Recording, "get_recording", _fake_get_recording)

    try:
        with pytest.raises(ValueError, match="differing exact timestamps"):
            SharedArtifactGroup.create_group(
                "session_times_.nwb",
                "v2_a25_timestamp_mismatch",
                [{"recording_id": rid_a}, {"recording_id": rid_b}],
            )
        assert (
            len(
                SharedArtifactGroup
                & {"shared_artifact_group_name": "v2_a25_timestamp_mismatch"}
            )
            == 0
        )
    finally:
        _drop_fake_recording(rid_a)
        _drop_fake_recording(rid_b)
    # The equal-fingerprint branch is covered by
    # test_shared_artifact_group_insert_rejects_mismatched_dtypes.


@pytest.mark.usefixtures("dj_conn")
def test_artifact_detection_selection_missing_lookup_row_diagnostic():
    """A missing ``ArtifactDetectionParameters`` row raises a ValueError naming
    the Lookup table and ``insert_default()``, not an FK IntegrityError."""
    import uuid

    from spyglass.spikesorting.v2.artifact import RecordingArtifactSelection

    with pytest.raises(ValueError) as excinfo:
        RecordingArtifactSelection.insert_selection(
            {
                "recording_id": uuid.uuid4(),
                "artifact_detection_params_name": "v2_a25_no_such_params_row",
            }
        )
    message = str(excinfo.value)
    assert "ArtifactDetectionParameters" in message
    assert "insert_default" in message


@pytest.mark.usefixtures("dj_conn")
def test_detect_artifacts_empty_sliver_filter_returns_empty():
    """When ``min_length_s`` filters out every kept interval,
    ``detect_artifacts`` returns a (0, 2) float array, not an empty 1-D one."""
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    rec = _synthetic_artifact_recording()
    validated = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,  # trips on the planted 300 µV burst
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=1.0,
        join_window_ms=0.0,
        min_length_s=1e9,  # larger than the whole recording -> all filtered
    )

    out = detect_artifacts(rec, validated)
    assert out.shape == (
        0,
        2,
    ), f"expected an empty (0, 2) array, got shape {out.shape}"
    # The downstream mask walker needs the timestamp dtype.
    assert out.dtype == rec.get_times().dtype


@pytest.mark.usefixtures("dj_conn")
def test_artifact_detection_delete_tolerates_already_gone_interval_list():
    """``RecordingArtifactDetection.delete`` does not raise when the paired
    IntervalList rows were already removed.

    Uses a dedicated planted selection so the shared fixture is untouched.
    """
    import uuid

    import datajoint as dj

    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2._artifacts.naming import (
        artifact_detection_interval_list_name,
    )

    nwb = "a25_delete_session_.nwb"
    rec_id = _plant_fake_recording(uuid.uuid4(), nwb, 30000.0)
    aid = uuid.uuid4()
    params_name = "v2_a25_delete_params"
    ilist_name = artifact_detection_interval_list_name(aid)
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        RecordingArtifactSelection.insert1(
            {
                "artifact_detection_id": aid,
                "artifact_detection_params_name": params_name,
                "recording_id": rec_id,
            },
            allow_direct_insert=True,
        )
        RecordingArtifactDetection.insert1(
            {"artifact_detection_id": aid}, allow_direct_insert=True
        )
        # The IntervalList row the delete would resolve and try to clean.
        IntervalList.insert1(
            {
                "nwb_file_name": nwb,
                "interval_list_name": ilist_name,
                "valid_times": np.empty((0, 2)),
            },
            allow_direct_insert=True,
        )
        RecordingArtifactDetection.RemovedInterval.insert1(
            {
                "artifact_detection_id": aid,
                "nwb_file_name": nwb,
                "interval_list_name": ilist_name,
            }
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        # Delete the IntervalList row out of band, keeping the ownership part.
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (
                IntervalList
                & {
                    "nwb_file_name": nwb,
                    "interval_list_name": ilist_name,
                }
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")

        # Must not raise.
        (RecordingArtifactDetection & {"artifact_detection_id": aid}).delete(
            safemode=False
        )
        assert (
            len(RecordingArtifactDetection & {"artifact_detection_id": aid})
            == 0
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (
                RecordingArtifactDetection & {"artifact_detection_id": aid}
            ).delete_quick()
            (
                RecordingArtifactSelection & {"artifact_detection_id": aid}
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        _drop_fake_recording(rec_id)


@pytest.mark.usefixtures("dj_conn")
def test_artifact_detection_delete_refuses_master_without_part_rows():
    """Deleting a ``RecordingArtifactDetection`` master with no
    ``RemovedInterval`` part row raises rather than guessing interval
    ownership, and leaves the master in place."""
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )

    nwb = "ownership_nopart_session_.nwb"
    rec_id = _plant_fake_recording(uuid.uuid4(), nwb, 30000.0)
    aid = uuid.uuid4()
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        RecordingArtifactSelection.insert1(
            {
                "artifact_detection_id": aid,
                "artifact_detection_params_name": "v2_ownership_nopart_params",
                "recording_id": rec_id,
            },
            allow_direct_insert=True,
        )
        RecordingArtifactDetection.insert1(
            {"artifact_detection_id": aid}, allow_direct_insert=True
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        with pytest.raises(ValueError, match="Refusing to guess"):
            (
                RecordingArtifactDetection & {"artifact_detection_id": aid}
            ).delete(safemode=False)
        assert (
            len(RecordingArtifactDetection & {"artifact_detection_id": aid})
            == 1
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (
                RecordingArtifactDetection & {"artifact_detection_id": aid}
            ).delete_quick()
            (
                RecordingArtifactSelection & {"artifact_detection_id": aid}
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        _drop_fake_recording(rec_id)


@pytest.mark.usefixtures("dj_conn")
def test_shared_group_membership_is_fixed_and_frozen(monkeypatch):
    """A group's members cannot change after creation, and a detection reads
    and masks only the members frozen with its selection.

    A member added through the maintenance bypass reaches neither the
    detection's inputs nor the coverage check.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2._storage.nwb import StoredTraces
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        SharedArtifactDetection,
        SharedArtifactGroup,
        SharedArtifactSelection,
        assert_artifact_detection_covers_recording,
    )
    from spyglass.spikesorting.v2.exceptions import (
        ImmutableRowError,
        UnsupportedDirectInsertError,
    )
    from spyglass.spikesorting.v2.recording import Recording

    ArtifactDetectionParameters.insert_default()
    session = "v2_member_freeze_.nwb"
    rid_a = _plant_fake_recording(uuid.uuid4(), session, 30000.0)
    rid_b = _plant_fake_recording(uuid.uuid4(), session, 30000.0)
    rid_c = _plant_fake_recording(uuid.uuid4(), session, 30000.0)
    group_key = {
        "nwb_file_name": session,
        "shared_artifact_group_name": "v2_member_freeze_group",
    }
    times = np.arange(8, dtype=np.float64) / 30000.0
    monkeypatch.setattr(
        Recording, "get_recording", lambda self, key: _FakeRecording(times)
    )
    monkeypatch.setattr(
        Recording,
        "resolve_stored_traces",
        lambda self, key: StoredTraces(
            f"/fake/{key['recording_id']}.nwb", "/fake/es", "0" * 64
        ),
    )
    conn = dj.conn()
    art_pk = None
    # The planted session has no Session row.
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SharedArtifactGroup.create_group(
            session,
            group_key["shared_artifact_group_name"],
            [{"recording_id": rid_a}, {"recording_id": rid_b}],
        )
        # The same members again is a no-op; other members are refused.
        SharedArtifactGroup.create_group(
            session,
            group_key["shared_artifact_group_name"],
            [{"recording_id": rid_b}, {"recording_id": rid_a}],
        )
        with pytest.raises(ValueError, match="exists with other members"):
            SharedArtifactGroup.create_group(
                session,
                group_key["shared_artifact_group_name"],
                [{"recording_id": rid} for rid in (rid_a, rid_b, rid_c)],
            )
        with pytest.raises(UnsupportedDirectInsertError, match="create_group"):
            SharedArtifactGroup.Member.insert1(
                {**group_key, "recording_id": rid_c}
            )
        with pytest.raises(UnsupportedDirectInsertError, match="create_group"):
            SharedArtifactGroup.insert1(
                {**group_key, "shared_artifact_group_name": "v2_other"}
            )
        with pytest.raises(ImmutableRowError):
            SharedArtifactGroup().update1(group_key)

        art_pk = SharedArtifactSelection.insert_selection(
            {**group_key, "artifact_detection_params_name": "default"}
        )
        frozen = {str(rid_a), str(rid_b)}
        assert {
            str(r)
            for r in (SharedArtifactSelection.Member & art_pk).fetch(
                "recording_id"
            )
        } == frozen

        # A maintenance bypass adds a member after the selection.
        SharedArtifactGroup.Member.insert1(
            {**group_key, "recording_id": rid_c}, allow_direct_insert=True
        )
        fetched = SharedArtifactDetection().make_fetch(art_pk)
        assert set(fetched.member_recording_ids) == frozen
        SharedArtifactDetection.insert1(art_pk, allow_direct_insert=True)
        for rid in (rid_a, rid_b):
            assert_artifact_detection_covers_recording(
                recording_id=rid,
                artifact_detection_id=art_pk["artifact_detection_id"],
                caller="test",
            )
        with pytest.raises(ValueError, match="do not include"):
            assert_artifact_detection_covers_recording(
                recording_id=rid_c,
                artifact_detection_id=art_pk["artifact_detection_id"],
                caller="test",
            )
    finally:
        try:
            if art_pk is not None:
                for table in (
                    SharedArtifactDetection,
                    SharedArtifactSelection.Member,
                    SharedArtifactSelection,
                ):
                    (table & art_pk).delete_quick()
            (SharedArtifactGroup.Member & group_key).delete_quick()
            (SharedArtifactGroup & group_key).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        for rid in (rid_a, rid_b, rid_c):
            _drop_fake_recording(rid)


def _orphan_masters():
    """Shared-artifact masters left without their part rows."""
    from spyglass.spikesorting.v2.artifact import (
        SharedArtifactDetection,
        SharedArtifactGroup,
        SharedArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import (
        ArtifactDetectionOutput,
    )

    return {
        "group": len(SharedArtifactGroup - SharedArtifactGroup.Member),
        "selection": len(
            SharedArtifactSelection - SharedArtifactSelection.Member
        ),
        "detection": len(
            SharedArtifactDetection - SharedArtifactDetection.RemovedInterval
        ),
        "output": len(
            ArtifactDetectionOutput
            - ArtifactDetectionOutput.RecordingArtifactDetection
            - ArtifactDetectionOutput.SharedArtifactDetection
        ),
    }


@pytest.mark.slow
@pytest.mark.integration
def test_deleting_member_recording_removes_its_shared_detections(
    chronic_2_session_minirec, monkeypatch
):
    """Deleting a member ``Recording`` removes every group holding it, with
    each group's selection, populated detection, merge registration, interval
    ownership rows and the sorts that use it; groups without it stay whole.

    ``g12`` and ``g2`` share member ``r2``; ``g1`` holds only ``r1``.
    """
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2._artifacts.naming import (
        artifact_detection_interval_list_name,
    )
    from spyglass.spikesorting.v2._core.recipe_catalog import (
        CORTEX_PREPROC,
        HIPPOCAMPUS_PREPROC,
    )
    from spyglass.spikesorting.v2 import artifact as artifact_module
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        SharedArtifactDetection,
        SharedArtifactGroup,
        SharedArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import (
        ArtifactDetectionOutput,
    )
    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    member = chronic_2_session_minirec["same_day_members"][0]
    nwb_file_name = member["nwb_file_name"]
    window = "v2_shared_cascade_window"
    raw = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": window,
            "valid_times": np.asarray([[raw[0][0] + 0.5, raw[0][0] + 4.0]]),
            "pipeline": "shared_group_cascade_test",
        },
        skip_duplicates=True,
    )
    ArtifactDetectionParameters.insert_default()
    baseline = _orphan_masters()
    # Both members are the one tetrode under two recipes, so their contacts
    # coincide; the union scan is not what this test checks.
    monkeypatch.setattr(
        artifact_module.si,
        "aggregate_channels",
        lambda recordings: recordings[0],
    )
    r1, r2 = (
        RecordingSelection.insert_selection(
            {
                **member,
                "interval_list_name": window,
                "preprocessing_params_name": recipe,
                "team_name": chronic_2_session_minirec["owner"],
            }
        )
        for recipe in (HIPPOCAMPUS_PREPROC, CORTEX_PREPROC)
    )
    Recording.populate([r1, r2], reserve_jobs=False)

    groups = {"g12": [r1, r2], "g2": [r2], "g1": [r1]}
    detections, sorts = {}, {}
    for name, members in groups.items():
        SharedArtifactGroup.create_group(nwb_file_name, name, members)
        detections[name] = SharedArtifactSelection.insert_selection(
            {
                "nwb_file_name": nwb_file_name,
                "shared_artifact_group_name": name,
                "artifact_detection_params_name": "none",
            }
        )
        SharedArtifactDetection.populate(detections[name], reserve_jobs=False)
        sorts[name] = SortingSelection.insert_selection(
            {
                "recording_id": members[-1]["recording_id"],
                "sorter": "mountainsort5",
                "sorter_params_name": "franklab_30khz_ms5_2026_06",
                **detections[name],
            }
        )
    assert _orphan_masters() == baseline

    def _present(name):
        detection = detections[name]
        rows = {
            "group": SharedArtifactGroup
            & {
                "nwb_file_name": nwb_file_name,
                "shared_artifact_group_name": name,
            },
            "selection": SharedArtifactSelection & detection,
            "frozen members": SharedArtifactSelection.Member & detection,
            "detection": SharedArtifactDetection & detection,
            "interval owner": SharedArtifactDetection.RemovedInterval
            & detection,
            "merge": ArtifactDetectionOutput.SharedArtifactDetection
            & detection,
            "sort": SortingSelection & sorts[name],
        }
        return {label: bool(rows[label]) for label in rows}

    try:
        (Recording & r2).delete(safemode=False)
        for name in ("g12", "g2"):
            assert not any(_present(name).values()), (name, _present(name))
        assert all(_present("g1").values()), _present("g1")
        assert _orphan_masters() == baseline

        (Recording & r1).delete(safemode=False)
        assert not any(_present("g1").values()), _present("g1")
        assert _orphan_masters() == baseline
    finally:
        (RecordingSelection & [r1, r2]).delete(safemode=False)
        # An upstream cascade leaves a detection's IntervalList row behind.
        (
            IntervalList
            & {"nwb_file_name": nwb_file_name}
            & [
                {
                    "interval_list_name": artifact_detection_interval_list_name(
                        pk["artifact_detection_id"]
                    )
                }
                for pk in detections.values()
            ]
            + [{"interval_list_name": window}]
        ).delete_quick()


@pytest.mark.usefixtures("dj_conn")
def test_importing_artifact_declares_the_detection_output_merge(tmp_path):
    """Importing ``artifact`` alone declares ``ArtifactDetectionOutput``.

    DataJoint cannot create a table inside the populate transaction where
    ``make_insert`` registers the merge row.
    """
    import os
    import subprocess
    import sys

    import datajoint as dj

    env = {
        **os.environ,
        "DJ_HOST": f"{dj.config['database.host']}:{dj.config['database.port']}",
        "DJ_USER": dj.config["database.user"],
        "DJ_PASS": dj.config["database.password"],
    }
    code = (
        "import sys\n"
        "import spyglass.spikesorting.v2.artifact\n"
        "print('spyglass.spikesorting.v2.artifact_output' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "True"
