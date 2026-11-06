"""Artifact-detection invariants: shared-group guards and selection branches.

The insert_group checks fire before any recording load, so preconditions are
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
    insert_group guards read. Returns the ``recording_id``.
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

    Implements the SI calls ``insert_group`` uses (``get_time_info``,
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
            SharedArtifactGroup.insert_group(
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
            SharedArtifactGroup.insert_group(
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
            SharedArtifactGroup.insert_group(
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
def test_shared_group_member_set_frozen(monkeypatch):
    """A ``SharedArtifactGroup.Member`` added after selection makes
    ``make_fetch`` raise a member-drift error; an unedited selection fetches."""
    import uuid

    from spyglass.spikesorting.v2._storage.nwb import StoredTraces
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        SharedArtifactGroup,
        SharedGroupArtifactDetection,
        SharedGroupArtifactSelection,
    )
    from spyglass.spikesorting.v2.exceptions import (
        SharedArtifactGroupMemberDriftError,
    )
    from spyglass.spikesorting.v2.recording import Recording

    import datajoint as dj

    ArtifactDetectionParameters.insert_default()
    session = "v2_member_freeze_.nwb"
    rid_a = _plant_fake_recording(uuid.uuid4(), session, 30000.0)
    rid_b = _plant_fake_recording(uuid.uuid4(), session, 30000.0)
    rid_c = _plant_fake_recording(uuid.uuid4(), session, 30000.0)
    group_name = "v2_member_freeze_group"
    art_pk = None
    conn = dj.conn()
    # Bypass insert_group, which loads member recordings; the drift check
    # reads only recording_ids. The file lookup after it is stubbed.
    monkeypatch.setattr(
        Recording,
        "resolve_stored_traces",
        lambda self, key: StoredTraces(
            f"/fake/{key['recording_id']}.nwb", "/fake/es", "0" * 64
        ),
    )
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SharedArtifactGroup.insert1(
            {
                "shared_artifact_group_name": group_name,
                "nwb_file_name": session,
            },
            allow_direct_insert=True,
        )
        SharedArtifactGroup.Member.insert(
            [
                {
                    "shared_artifact_group_name": group_name,
                    "recording_id": rid_a,
                },
                {
                    "shared_artifact_group_name": group_name,
                    "recording_id": rid_b,
                },
            ],
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        art_pk = SharedGroupArtifactSelection.insert_selection(
            {
                "shared_artifact_group_name": group_name,
                "artifact_detection_params_name": "default",
            }
        )
        # Unedited: the live member set matches the snapshot.
        fetched = SharedGroupArtifactDetection().make_fetch(art_pk)
        assert [t.abs_path for t in fetched.member_traces] == [
            f"/fake/{rid}.nwb" for rid in fetched.member_recording_ids
        ]

        # Drift: add a member after the snapshot was taken.
        SharedArtifactGroup.Member.insert1(
            {"shared_artifact_group_name": group_name, "recording_id": rid_c}
        )
        with pytest.raises(
            SharedArtifactGroupMemberDriftError, match="member set"
        ):
            SharedGroupArtifactDetection().make_fetch(art_pk)
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            if art_pk is not None:
                (SharedGroupArtifactSelection & art_pk).delete_quick()
            (
                SharedArtifactGroup.Member
                & {"shared_artifact_group_name": group_name}
            ).delete_quick()
            (
                SharedArtifactGroup & {"shared_artifact_group_name": group_name}
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        for rid in (rid_a, rid_b, rid_c):
            _drop_fake_recording(rid)


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
