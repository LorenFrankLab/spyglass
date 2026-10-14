"""Session cleanup respects real artifact-interval ownership across cascades."""

import uuid

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import _clean_session_v2


@pytest.fixture
def isolated_interval_owners(dj_conn):
    """Plant isolated relational metadata; no recording loads are needed.

    The deliberately shared ownership rows model direct inserts or repairs.
    They exercise both surviving owner tables after the target session's
    ordinary master/part cascades. Foreign keys are disabled only while
    planting fake upstream recording/session metadata and during teardown;
    the cleanup under test runs with foreign-key enforcement enabled.
    """
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
        SharedArtifactGroup,
        SharedGroupArtifactDetection,
        SharedGroupArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection

    token = uuid.uuid4().hex
    target = f"helper_cleanup_target_{token}_.nwb"
    survivor = f"helper_cleanup_survivor_{token}_.nwb"
    sessions = [{"nwb_file_name": target}, {"nwb_file_name": survivor}]
    recordings = [uuid.uuid4(), uuid.uuid4()]
    split_ids = [uuid.uuid4(), uuid.uuid4()]
    shared_ids = [uuid.uuid4(), uuid.uuid4()]
    group_names = [f"helper_target_{token}", f"helper_survivor_{token}"]
    recording_keys = [{"recording_id": value} for value in recordings]
    detection_keys = [
        {"artifact_detection_id": value} for value in split_ids + shared_ids
    ]
    group_keys = [
        {"shared_artifact_group_name": value} for value in group_names
    ]
    # Repeated names in a different session are intentional: cleanup must use
    # the full IntervalList key, and must not infer ownership from names.
    interval_keys = [
        {"nwb_file_name": target, "interval_list_name": name}
        for name in (
            "single-owned",
            "shared-owned",
            "remaining-shared-owner",
            "remaining-single-owner",
            "manual-unowned",
        )
    ] + [
        {"nwb_file_name": survivor, "interval_list_name": name}
        for name in ("single-owned", "shared-owned")
    ]
    expected_times = [
        np.array([[index + 1.0, index + 1.5]]) for index in range(7)
    ]

    try:
        dj_conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            for recording_id, session in zip(recordings, sessions):
                RecordingSelection.insert1(
                    {
                        "recording_id": recording_id,
                        **session,
                        "sort_group_id": 0,
                        "interval_list_name": "raw data valid times",
                        "preprocessing_params_name": "unused_cleanup_recipe",
                        "team_name": "unused_cleanup_team",
                    },
                    allow_direct_insert=True,
                )
                Recording.insert1(
                    {
                        "recording_id": recording_id,
                        "analysis_file_name": f"unused_cleanup_{token}.nwb",
                        "electrical_series_path": "/unused",
                        "object_id": "unused-cleanup-object",
                        "n_channels": 4,
                        "sampling_frequency": 1000.0,
                        "duration_s": 10.0,
                        "content_hash": "0" * 64,
                    },
                    allow_direct_insert=True,
                )
            for index, session in enumerate(sessions):
                RecordingArtifactSelection.insert1(
                    {
                        "artifact_detection_id": split_ids[index],
                        "recording_id": recordings[index],
                        "artifact_detection_params_name": "unused_cleanup_recipe",
                    },
                    allow_direct_insert=True,
                )
                RecordingArtifactDetection.insert1(
                    {"artifact_detection_id": split_ids[index]},
                    allow_direct_insert=True,
                )
                SharedArtifactGroup.insert1({**group_keys[index], **session})
                SharedArtifactGroup.Member.insert1(
                    {**group_keys[index], **recording_keys[index]}
                )
                SharedGroupArtifactSelection.insert1(
                    {
                        "artifact_detection_id": shared_ids[index],
                        **group_keys[index],
                        "artifact_detection_params_name": "unused_cleanup_recipe",
                        "member_set_hash": "0" * 64,
                    },
                    allow_direct_insert=True,
                )
                SharedGroupArtifactDetection.insert1(
                    {"artifact_detection_id": shared_ids[index]},
                    allow_direct_insert=True,
                )
            IntervalList.insert(
                [
                    {**key, "valid_times": times}
                    for key, times in zip(interval_keys, expected_times)
                ]
            )
            RecordingArtifactDetection.RemovedInterval.insert(
                [
                    {"artifact_detection_id": split_ids[0], **interval_keys[0]},
                    {"artifact_detection_id": split_ids[0], **interval_keys[2]},
                    {"artifact_detection_id": split_ids[1], **interval_keys[3]},
                    {"artifact_detection_id": split_ids[1], **interval_keys[5]},
                ]
            )
            SharedGroupArtifactDetection.RemovedInterval.insert(
                [
                    {
                        "artifact_detection_id": shared_ids[0],
                        **interval_keys[1],
                    },
                    {
                        "artifact_detection_id": shared_ids[0],
                        **interval_keys[3],
                    },
                    {
                        "artifact_detection_id": shared_ids[1],
                        **interval_keys[2],
                    },
                ]
            )
        finally:
            dj_conn.query("SET FOREIGN_KEY_CHECKS=1")
        yield {
            "target": sessions[0],
            "survivor": sessions[1],
            "recordings": recording_keys,
            "groups": group_keys,
            "split_ids": split_ids,
            "shared_ids": shared_ids,
            "interval_keys": interval_keys,
            "expected_times": expected_times,
        }
    finally:
        # Teardown does not call the helper being tested: a failing helper
        # cannot leave its planted rows to contaminate later tests.
        dj_conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            for table in (
                RecordingArtifactDetection.RemovedInterval,
                SharedGroupArtifactDetection.RemovedInterval,
                RecordingArtifactDetection,
                SharedGroupArtifactDetection,
                RecordingArtifactSelection,
                SharedGroupArtifactSelection,
            ):
                (table & detection_keys).delete_quick()
            (SharedArtifactGroup.Member & group_keys).delete_quick()
            (SharedArtifactGroup & group_keys).delete_quick()
            (Recording & recording_keys).delete_quick()
            (RecordingSelection & recording_keys).delete_quick()
            (IntervalList & interval_keys).delete_quick()
        finally:
            dj_conn.query("SET FOREIGN_KEY_CHECKS=1")


@pytest.mark.database
@pytest.mark.integration
def test_clean_session_removes_only_abandoned_owned_intervals(
    isolated_interval_owners,
):
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        SharedArtifactGroup,
        SharedGroupArtifactDetection,
    )
    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection

    graph = isolated_interval_owners
    interval_keys = graph["interval_keys"]
    assert len(IntervalList & interval_keys) == 7
    for _ in range(2):
        _clean_session_v2(graph["target"])
        assert not (RecordingSelection & graph["target"])
        assert not (Recording & graph["recordings"][0])
        assert not (SharedArtifactGroup & graph["groups"][0])
        assert not (
            RecordingArtifactDetection
            & {"artifact_detection_id": graph["split_ids"][0]}
        )
        assert not (
            SharedGroupArtifactDetection
            & {"artifact_detection_id": graph["shared_ids"][0]}
        )
        for key in interval_keys[:2]:
            assert not (IntervalList & key), key
        assert len(IntervalList & interval_keys) == 5
        for key, times in zip(interval_keys[2:], graph["expected_times"][2:]):
            np.testing.assert_array_equal(
                (IntervalList & key).fetch1("valid_times"), times
            )
        assert RecordingSelection & graph["survivor"]
        assert Recording & graph["recordings"][1]
        assert SharedArtifactGroup & graph["groups"][1]
        assert (
            SharedGroupArtifactDetection.RemovedInterval
            & {"artifact_detection_id": graph["shared_ids"][1]}
            & interval_keys[2]
        )
        assert (
            RecordingArtifactDetection.RemovedInterval
            & {"artifact_detection_id": graph["split_ids"][1]}
            & interval_keys[3]
        )
