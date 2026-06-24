"""Artifact-detection invariants: shared-group guards and selection branches.

Covers ``SharedArtifactGroup.insert_group`` cross-session / cross-frequency /
timestamp guards, ``ArtifactDetectionSelection.insert_selection`` duplicate +
missing-Lookup-row diagnostics, the empty sliver-filter return, the
ArtifactRemovedInterval ownership invariants, and tolerant delete of an
already-gone IntervalList.

The insert_group session/frequency checks and the duplicate check all fire
BEFORE any recording load, so every precondition is built via a contained
FK-checks-off raw bypass -- no real populate needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import _synthetic_artifact_recording


def _plant_fake_recording(recording_id, nwb_file_name, sampling_frequency):
    """Insert a minimal ``Recording`` + ``RecordingSelection`` pair via the
    FK-checks-off bypass.

    ``SharedArtifactGroup.insert_group`` reads ``RecordingSelection.
    nwb_file_name`` (session check) and ``Recording.sampling_frequency``
    (frequency check) and only requires the ``Recording`` row to *exist*
    for the populated-check -- all before it ever loads the recording. So
    a fake pair with the right two scalar fields is enough to drive both
    guards without a real populate. Returns the ``recording_id``.
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
                "cache_hash": "0" * 64,
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


@pytest.mark.usefixtures("dj_conn")
def test_shared_artifact_group_rejects_cross_session_members():
    """Members from two sessions raise, and no master row is left.

    A shared artifact pass writes IntervalList rows keyed by
    ``(nwb_file_name, interval_list_name)``, so mixing sessions makes the
    artifact-removed valid times undefined. The guard fires before the
    master insert, so the table must be empty for this group name after
    the raise (transactional rollback / pre-insert raise).
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

    ``si.aggregate_channels`` requires identical fs; a typo in upstream
    preproc could otherwise produce a mismatch that only crashes opaquely
    deep inside SI at populate time. Both members share one session so the
    session guard passes and the frequency guard is the one that fires.
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

    The two fake recordings share session, sampling frequency, sample count,
    and dtype. Only their wall-clock vectors differ. Without the exact
    timestamp guard, ``si.aggregate_channels`` would stack non-aligned frames
    and the resulting artifact intervals could mask the wrong times.
    """
    import uuid

    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup
    from spyglass.spikesorting.v2.recording import Recording

    rid_a = _plant_fake_recording(uuid.uuid4(), "session_times_.nwb", 30000.0)
    rid_b = _plant_fake_recording(uuid.uuid4(), "session_times_.nwb", 30000.0)

    class _FakeRecording:
        """Faithful stand-in for an explicit-time_vector SI recording.

        Models the SI public API ``SharedArtifactGroup.insert_group`` uses --
        ``get_time_info`` (lazy time_vector), ``sample_index_to_time`` (the
        chunked fingerprint reads), and segment-indexed ``get_num_samples`` -- so
        the test exercises the real ``timestamp_fingerprint`` path rather than a
        frame-bounded shortcut real SI lacks.
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


@pytest.mark.usefixtures("dj_conn")
def test_artifact_detection_selection_raises_duplicate_selection_error():
    """Two master rows for one (params, source) raise
    ``DuplicateSelectionError``.

    ``insert_selection`` is find-existing-or-insert; if a prior direct
    bypass left two masters sharing the same ``artifact_detection_params_name`` and
    ``recording_id``, the find step sees >1 and must raise the integrity
    error rather than silently picking one. Plant the duplicate masters +
    matching ``RecordingSource`` parts via the FK-checks-off bypass (the
    only way to land the otherwise-impossible state).
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.artifact import ArtifactDetectionSelection
    from spyglass.spikesorting.v2.exceptions import DuplicateSelectionError

    params_name = "v2_a25_dup_params"
    rec_id = uuid.uuid4()
    aid1, aid2 = uuid.uuid4(), uuid.uuid4()
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        for aid in (aid1, aid2):
            ArtifactDetectionSelection.insert1(
                {
                    "artifact_detection_id": aid,
                    "artifact_detection_params_name": params_name,
                },
                allow_direct_insert=True,
            )
            ArtifactDetectionSelection.RecordingSource.insert1(
                {"artifact_detection_id": aid, "recording_id": rec_id},
                allow_direct_insert=True,
            )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        with pytest.raises(DuplicateSelectionError, match="master rows"):
            ArtifactDetectionSelection.insert_selection(
                {
                    "recording_id": rec_id,
                    "artifact_detection_params_name": params_name,
                }
            )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (
                ArtifactDetectionSelection.RecordingSource
                & {"recording_id": rec_id}
            ).delete_quick()
            for aid in (aid1, aid2):
                (
                    ArtifactDetectionSelection & {"artifact_detection_id": aid}
                ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")


@pytest.mark.usefixtures("dj_conn")
def test_artifact_detection_selection_requires_artifact_detection_params_name():
    """A key without ``artifact_detection_params_name`` raises naming the field.

    The source key alone is insufficient -- the master row needs the
    params FK. The guard names the missing field so the notebook user can
    fix the call in one step.
    """
    import uuid

    from spyglass.spikesorting.v2.artifact import ArtifactDetectionSelection

    with pytest.raises(ValueError, match="artifact_detection_params_name"):
        ArtifactDetectionSelection.insert_selection(
            {"recording_id": uuid.uuid4()}
        )


@pytest.mark.usefixtures("dj_conn")
def test_artifact_detection_selection_missing_lookup_row_diagnostic():
    """A missing ``ArtifactDetectionParameters`` row raises a
    diagnostic ValueError, not an opaque FK IntegrityError.

    ``_ensure_lookup_row_exists`` pre-checks the params FK target and, when
    absent, names the Lookup table and the ``insert_default()`` path. The
    source key has no matching master yet (find returns empty), so control
    reaches the lookup pre-check.
    """
    import uuid

    from spyglass.spikesorting.v2.artifact import ArtifactDetectionSelection

    with pytest.raises(ValueError) as excinfo:
        ArtifactDetectionSelection.insert_selection(
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
    ``_detect_artifacts`` returns ``np.empty((0, 2))``.

    Detection finds artifact frames (so the early all-valid return at the
    zero-frames branch is NOT taken), the complement is built, then a huge
    ``min_length_s`` drops every surviving valid sliver. The return must be
    a real (0, 2) float array -- the shape downstream ``_apply_artifact_
    mask`` expects -- not an empty 1-D array. Pairs with the
    ``_apply_artifact_mask`` empty-valid-times raise.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

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

    out = ArtifactDetection._detect_artifacts(rec, validated)
    assert out.shape == (
        0,
        2,
    ), f"expected an empty (0, 2) array, got shape {out.shape}"
    # dtype matches the recording's timestamp dtype (float64); a 1-D
    # np.empty(0) or an int array would break the downstream mask walker.
    assert out.dtype == rec.get_times().dtype


@pytest.mark.usefixtures("dj_conn")
def test_artifact_detection_delete_tolerates_already_gone_interval_list():
    """``ArtifactDetection.delete`` does not raise when the paired
    IntervalList rows were already removed.

    The override deletes the master then cleans up the IntervalList rows it
    resolved; the ``len(rows) == 0`` guard skips a restriction whose rows
    are already gone. We pre-delete the IntervalList rows, then delete the
    detection, and assert no raise + the master is gone.

    Builds a DEDICATED selection on a planted recording (FK-off) and
    inserts a zero-row ``ArtifactDetection`` master directly, so the shared
    populated fixture's artifact row is untouched.
    """
    import uuid

    import datajoint as dj

    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.utils import (
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
        ArtifactDetectionSelection.insert1(
            {
                "artifact_detection_id": aid,
                "artifact_detection_params_name": params_name,
            },
            allow_direct_insert=True,
        )
        ArtifactDetectionSelection.RecordingSource.insert1(
            {"artifact_detection_id": aid, "recording_id": rec_id},
            allow_direct_insert=True,
        )
        ArtifactDetection.insert1(
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
        ArtifactDetection.ArtifactRemovedInterval.insert1(
            {
                "artifact_detection_id": aid,
                "nwb_file_name": nwb,
                "interval_list_name": ilist_name,
            }
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        # Pre-delete the IntervalList row so delete() hits the already-gone
        # (len == 0) branch. Disable FK checks here so the ownership part row
        # remains; this models an out-of-band broken interval row while
        # preserving the strict ArtifactRemovedInterval ownership invariant.
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

        # Must not raise even though the cleanup target is already gone.
        (ArtifactDetection & {"artifact_detection_id": aid}).delete(
            safemode=False
        )
        assert len(ArtifactDetection & {"artifact_detection_id": aid}) == 0
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (ArtifactDetection & {"artifact_detection_id": aid}).delete_quick()
            (
                ArtifactDetectionSelection.RecordingSource
                & {"artifact_detection_id": aid}
            ).delete_quick()
            (
                ArtifactDetectionSelection & {"artifact_detection_id": aid}
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        _drop_fake_recording(rec_id)


def test_artifact_detection_delete_refuses_master_without_part_rows():
    """Strict ownership: deleting an ``ArtifactDetection`` master that owns NO
    ``ArtifactRemovedInterval`` part row raises rather than guessing interval
    ownership from naming -- the data-loss guard the part-table refactor adds.
    The refused delete must leave the master in place.

    Builds a DEDICATED selection on a planted recording (FK-off) with a
    zero-row master and NO part row, so the shared populated fixture is
    untouched.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
    )

    nwb = "ownership_nopart_session_.nwb"
    rec_id = _plant_fake_recording(uuid.uuid4(), nwb, 30000.0)
    aid = uuid.uuid4()
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        ArtifactDetectionSelection.insert1(
            {
                "artifact_detection_id": aid,
                "artifact_detection_params_name": "v2_ownership_nopart_params",
            },
            allow_direct_insert=True,
        )
        ArtifactDetectionSelection.RecordingSource.insert1(
            {"artifact_detection_id": aid, "recording_id": rec_id},
            allow_direct_insert=True,
        )
        ArtifactDetection.insert1(
            {"artifact_detection_id": aid}, allow_direct_insert=True
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        with pytest.raises(ValueError, match="Refusing to guess"):
            (ArtifactDetection & {"artifact_detection_id": aid}).delete(
                safemode=False
            )
        assert len(ArtifactDetection & {"artifact_detection_id": aid}) == 1
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (ArtifactDetection & {"artifact_detection_id": aid}).delete_quick()
            (
                ArtifactDetectionSelection.RecordingSource
                & {"artifact_detection_id": aid}
            ).delete_quick()
            (
                ArtifactDetectionSelection & {"artifact_detection_id": aid}
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        _drop_fake_recording(rec_id)


def test_read_artifact_removed_intervals_rejects_multiple_recording_part_rows():
    """A recording-backed ``ArtifactDetection`` must own exactly one
    ``IntervalList`` row; two ownership part rows is a corrupted state, and the
    read path raises rather than silently picking one.
    """
    import uuid

    import datajoint as dj

    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2._artifact_intervals import (
        read_artifact_removed_intervals,
    )
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
    )

    nwb = "ownership_twopart_session_.nwb"
    rec_id = _plant_fake_recording(uuid.uuid4(), nwb, 30000.0)
    aid = uuid.uuid4()
    inames = [f"artifact_detection_{aid}_{s}" for s in ("a", "b")]
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        ArtifactDetectionSelection.insert1(
            {
                "artifact_detection_id": aid,
                "artifact_detection_params_name": "v2_ownership_twopart_params",
            },
            allow_direct_insert=True,
        )
        ArtifactDetectionSelection.RecordingSource.insert1(
            {"artifact_detection_id": aid, "recording_id": rec_id},
            allow_direct_insert=True,
        )
        ArtifactDetection.insert1(
            {"artifact_detection_id": aid}, allow_direct_insert=True
        )
        for iname in inames:
            IntervalList.insert1(
                {
                    "nwb_file_name": nwb,
                    "interval_list_name": iname,
                    "valid_times": np.empty((0, 2)),
                },
                allow_direct_insert=True,
            )
            ArtifactDetection.ArtifactRemovedInterval.insert1(
                {
                    "artifact_detection_id": aid,
                    "nwb_file_name": nwb,
                    "interval_list_name": iname,
                }
            )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        with pytest.raises(ValueError, match="expected exactly one"):
            read_artifact_removed_intervals({"artifact_detection_id": aid})
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (
                ArtifactDetection.ArtifactRemovedInterval
                & {"artifact_detection_id": aid}
            ).delete_quick()
            (ArtifactDetection & {"artifact_detection_id": aid}).delete_quick()
            (
                ArtifactDetectionSelection.RecordingSource
                & {"artifact_detection_id": aid}
            ).delete_quick()
            (
                ArtifactDetectionSelection & {"artifact_detection_id": aid}
            ).delete_quick()
            for iname in inames:
                (
                    IntervalList
                    & {"nwb_file_name": nwb, "interval_list_name": iname}
                ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        _drop_fake_recording(rec_id)
