"""Failure translation of the shared artifact-bound selection insert.

``SortingSelection`` and ``MotionEstimateSelection`` link an optional
artifact detection through one insert protocol. A detection that is not
materialized and an input foreign key that fails mid-insert (a raw insert or
a concurrent delete) must surface as ``SchemaBypassError`` naming the
selection table, not as a bare ``KeyError`` / ``IntegrityError``. These
tests drive the motion-estimate path directly, since the public
``insert_selection`` checks the detection and source exist first.
"""

from __future__ import annotations

import uuid

import pytest


def _master_row(motion_estimate_id: uuid.UUID) -> dict:
    return {
        "motion_estimate_id": motion_estimate_id,
        "motion_estimation_params_name": "dredge_v1",
        "resolved_params_hash": "0" * 64,
        "spikeinterface_version": "0.104.3",
        "motion_algorithm_version": 1,
        "source_content_hash": "0" * 64,
    }


def test_motion_selection_unmaterialized_detection_is_schema_bypass(dj_conn):
    """A detection that cannot be registered is a SchemaBypassError."""
    from spyglass.spikesorting.v2._motion.selection import (
        insert_estimate_selection_rows,
    )
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    estimate_id = uuid.uuid4()
    missing_detection = uuid.uuid4()
    with pytest.raises(
        SchemaBypassError,
        match=(
            f"MotionEstimateSelection: artifact_detection_id "
            f"{missing_detection} is not materialized"
        ),
    ) as raised:
        insert_estimate_selection_rows(
            MotionEstimateSelection,
            _master_row(estimate_id),
            MotionEstimateSelection.RecordingSource,
            {"motion_estimate_id": estimate_id, "recording_id": uuid.uuid4()},
            missing_detection,
            refetch=lambda: None,
        )
    assert isinstance(raised.value.__cause__, KeyError)
    assert not (MotionEstimateSelection & {"motion_estimate_id": estimate_id})


def test_motion_selection_unsatisfied_source_fk_is_schema_bypass(dj_conn):
    """A source part whose Recording is missing is a SchemaBypassError, and
    the transaction leaves no master behind."""
    import datajoint as dj

    from spyglass.spikesorting.v2._motion.selection import (
        insert_estimate_selection_rows,
    )
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.motion import (
        MotionEstimateSelection,
        MotionEstimationParameters,
    )

    MotionEstimationParameters.insert_default()
    estimate_id = uuid.uuid4()
    missing_recording = uuid.uuid4()
    with pytest.raises(
        SchemaBypassError,
        match="MotionEstimateSelection: an input foreign key is unsatisfied",
    ) as raised:
        insert_estimate_selection_rows(
            MotionEstimateSelection,
            _master_row(estimate_id),
            MotionEstimateSelection.RecordingSource,
            {
                "motion_estimate_id": estimate_id,
                "recording_id": missing_recording,
            },
            None,
            refetch=lambda: None,
        )
    assert isinstance(raised.value.__cause__, dj.errors.IntegrityError)
    assert str(missing_recording) in str(raised.value)
    assert not (MotionEstimateSelection & {"motion_estimate_id": estimate_id})
