"""DB tests for the v2 motion-estimation tables.

The populate tests ingest a 30 s one-shank polymer session with a planted
+/-25 um rigid drift (``_motion_fixtures.write_drifting_polymer_nwb``),
populate its ``Recording`` and estimate on it. The estimator's accuracy is
covered DB-free in ``test_motion_estimation.py``; these tests cover the
tables: identity, guards, persistence and the compute-time checks.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

MOTION_TEAM = "motion_estimate_team"
DRIFT_NWB = "motion_drift_polymer.nwb"
DRIFT_DURATION_S = 30.0


@pytest.fixture
def motion_params(dj_conn):
    """``MotionEstimationParameters`` with the shipped rows installed."""
    from spyglass.spikesorting.v2.motion import MotionEstimationParameters

    MotionEstimationParameters.insert_default()
    return MotionEstimationParameters


def test_default_estimation_rows_install_idempotently(motion_params):
    motion_params.insert_default()

    names = set(motion_params.fetch("motion_estimation_params_name"))
    assert {"dredge_v1", "dredge_fast_v1"} <= names
    assert not (motion_params & {"motion_estimation_params_name": "rigid_fast"})
    presets = {
        name: params["preset"]
        for name, params in zip(
            *(
                motion_params
                & [
                    {"motion_estimation_params_name": "dredge_v1"},
                    {"motion_estimation_params_name": "dredge_fast_v1"},
                ]
            ).fetch("motion_estimation_params_name", "params")
        )
    }
    assert presets == {"dredge_v1": "dredge", "dredge_fast_v1": "dredge_fast"}


def test_estimation_row_with_unknown_override_is_rejected(motion_params):
    with pytest.raises(ValueError, match="detect_treshold"):
        motion_params.insert1(
            {
                "motion_estimation_params_name": "typo_row",
                "params": {
                    "preset": "dredge",
                    "detect_kwargs": {"detect_treshold": 6.0},
                },
            }
        )
    assert not (motion_params & {"motion_estimation_params_name": "typo_row"})


def test_estimation_row_rejects_job_kwargs_seed(motion_params):
    with pytest.raises(ValueError, match="noise_levels_seed"):
        motion_params.insert1(
            {
                "motion_estimation_params_name": "seeded_job_kwargs",
                "params": {"preset": "rigid_fast"},
                "job_kwargs": {"random_seed": 3},
            }
        )


def test_estimation_row_duplicating_a_default_is_rejected(motion_params):
    from spyglass.spikesorting.v2._params.motion_estimation import (
        MotionEstimationParamsSchema,
    )
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )

    with pytest.raises(DuplicateParameterContentError, match="dredge_v1"):
        motion_params.insert1(
            {
                "motion_estimation_params_name": "dredge_copy",
                "params": MotionEstimationParamsSchema(
                    preset="dredge"
                ).model_dump(),
            }
        )


def _drop_motion_selections(recording_key) -> None:
    """Delete every motion-estimate selection on a recording (masters first)."""
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    keys = (MotionEstimateSelection.RecordingSource & recording_key).fetch(
        "KEY", as_dict=True
    )
    if keys:
        (MotionEstimateSelection & keys).super_delete(
            warn=False, safemode=False
        )


@pytest.fixture(scope="module")
def drift_recording(dj_conn, tmp_path_factory):
    """A populated ``Recording`` of the planted-drift polymer session."""
    import datetime as dt

    from spyglass.spikesorting.v2.motion import MotionEstimationParameters
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        configure_v2_run_inputs,
        copy_and_insert_nwb,
    )
    from tests.spikesorting.v2._motion_fixtures import (
        write_drifting_polymer_nwb,
    )

    src = write_drifting_polymer_nwb(
        tmp_path_factory.mktemp("motion") / DRIFT_NWB,
        session_start=dt.datetime(2023, 6, 22, 12, tzinfo=dt.timezone.utc),
        fixture_name="motion_drift_polymer",
        seed=0,
        duration_s=DRIFT_DURATION_S,
    )
    nwb_file_name = copy_and_insert_nwb(src, dest_name=DRIFT_NWB)
    run = configure_v2_run_inputs(nwb_file_name, MOTION_TEAM)
    recording_key = RecordingSelection.insert_selection(
        {**run, "preprocessing_params_name": "default"}
    )
    _drop_motion_selections(recording_key)
    if not (Recording & recording_key):
        Recording.populate(recording_key, reserve_jobs=False)
    MotionEstimationParameters.insert_default()

    yield {"recording_key": recording_key, "nwb_file_name": nwb_file_name}

    _drop_motion_selections(recording_key)
    _clean_session_v2({"nwb_file_name": nwb_file_name})


def _select(recording_key, params_name="dredge_fast_v1", **extra):
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    return MotionEstimateSelection.insert_selection(
        {
            "recording_id": recording_key["recording_id"],
            "motion_estimation_params_name": params_name,
            **extra,
        }
    )


def test_selection_is_content_addressed_and_idempotent(drift_recording):
    import spikeinterface

    from spyglass.spikesorting.v2._motion import (
        MOTION_ALGORITHM_VERSION,
        motion_estimate_identity_payload,
        resolve_estimation_params,
        resolved_params_hash,
    )
    from spyglass.spikesorting.v2._selection_identity import deterministic_id
    from spyglass.spikesorting.v2.motion import (
        MotionEstimateSelection,
        MotionEstimationParameters,
    )
    from spyglass.spikesorting.v2.recording import Recording

    recording_key = drift_recording["recording_key"]
    first = _select(recording_key)
    again = _select({"recording_id": str(recording_key["recording_id"])})
    other_recipe = _select(recording_key, "dredge_v1")

    params = (
        MotionEstimationParameters
        & {"motion_estimation_params_name": "dredge_fast_v1"}
    ).fetch1("params")
    expected = deterministic_id(
        "motion_estimate",
        motion_estimate_identity_payload(
            source_kind="recording",
            source_id=recording_key["recording_id"],
            source_content_hash=(Recording & recording_key).fetch1(
                "content_hash"
            ),
            artifact_detection_id=None,
            motion_estimation_params_name="dredge_fast_v1",
            resolved_params_hash=resolved_params_hash(
                resolve_estimation_params(params)
            ),
            spikeinterface_version=spikeinterface.__version__,
            motion_algorithm_version=MOTION_ALGORITHM_VERSION,
        ),
    )
    assert first == again == {"motion_estimate_id": expected}
    assert other_recipe != first
    assert len(MotionEstimateSelection.RecordingSource & first) == 1
    assert not (MotionEstimateSelection.ConcatenatedRecordingSource & first)
    assert not (MotionEstimateSelection.ArtifactDetectionSource & first)
    lineage = MotionEstimateSelection.resolve_source(first)
    assert (lineage.kind, lineage.artifact_detection_id) == ("recording", None)

    with pytest.raises(ValueError, match="does not match the id derived"):
        _select(recording_key, motion_estimate_id=uuid.uuid4())


@pytest.mark.parametrize(
    "key, match",
    [
        ({"motion_estimation_params_name": "dredge_v1"}, "exactly one of"),
        (
            {
                "recording_id": uuid.uuid4(),
                "concat_recording_id": uuid.uuid4(),
                "motion_estimation_params_name": "dredge_v1",
            },
            "exactly one of",
        ),
        (
            {
                "concat_recording_id": uuid.uuid4(),
                "artifact_detection_id": uuid.uuid4(),
                "motion_estimation_params_name": "dredge_v1",
            },
            "own member artifact masks",
        ),
        (
            {
                "recording_id": uuid.uuid4(),
                "motion_estimation_params_name": "dredge_v1",
                "sorter": "mountainsort5",
            },
            "unknown field",
        ),
        (
            {
                "recording_id": uuid.uuid4(),
                "motion_estimation_params_name": "dredge_v1",
            },
            "Populate it",
        ),
    ],
    ids=["no-source", "two-sources", "concat-artifact", "extra", "unpopulated"],
)
def test_invalid_selection_is_rejected(motion_params, key, match):
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    with pytest.raises(ValueError, match=match):
        MotionEstimateSelection.insert_selection(key)


def test_live_concat_heading_is_current(dj_conn):
    """The recreated concat tables carry no removed motion column."""
    from spyglass.spikesorting.v2.motion import _assert_concat_tables_current

    _assert_concat_tables_current()


def test_direct_insert_and_orphan_master_are_refused(drift_recording):
    import datajoint as dj

    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    row = {
        "motion_estimate_id": uuid.uuid4(),
        "motion_estimation_params_name": "dredge_v1",
        "resolved_params_hash": "0" * 64,
        "spikeinterface_version": "0.104.3",
        "motion_algorithm_version": 1,
        "source_content_hash": "0" * 64,
    }
    with pytest.raises(dj.errors.DataJointError, match="insert_selection"):
        MotionEstimateSelection.insert1(row)

    MotionEstimateSelection.insert1(row, allow_direct_insert=True)
    try:
        with pytest.raises(SchemaBypassError, match="0 source part rows"):
            MotionEstimateSelection.resolve_source(row)
    finally:
        (
            MotionEstimateSelection
            & {"motion_estimate_id": row["motion_estimate_id"]}
        ).super_delete(warn=False, safemode=False)


def test_estimate_round_trip(drift_recording):
    """Populate persists the computed Motion exactly, with its resolved
    configuration, spans, geometry and diagnostics."""
    from unittest import mock

    from spyglass.spikesorting.v2 import _motion
    from spyglass.spikesorting.v2.motion import (
        MotionEstimate,
        MotionEstimateSelection,
        MotionEstimationParameters,
    )

    key = _select(drift_recording["recording_key"])
    computed = {}
    estimate_motion_in_spans = _motion.estimate_motion_in_spans

    def _observe(recording, **kwargs):
        motion, diagnostics = estimate_motion_in_spans(recording, **kwargs)
        computed.update(motion=motion, diagnostics=diagnostics)
        computed["locations"] = recording.get_channel_locations()
        return motion, diagnostics

    with mock.patch.object(_motion, "estimate_motion_in_spans", _observe):
        MotionEstimate.populate(key, reserve_jobs=False)

    row = (MotionEstimate & key).fetch1()
    stored = MotionEstimate().get_motion(key)
    motion = computed["motion"]
    np.testing.assert_array_equal(
        stored.displacement[0], motion.displacement[0]
    )
    np.testing.assert_array_equal(
        stored.temporal_bins_s[0], motion.temporal_bins_s[0]
    )
    np.testing.assert_array_equal(
        stored.spatial_bins_um, motion.spatial_bins_um
    )
    assert (stored.direction, stored.interpolation_method) == (
        motion.direction,
        motion.interpolation_method,
    )

    params = (
        MotionEstimationParameters
        & {"motion_estimation_params_name": "dredge_fast_v1"}
    ).fetch1("params")
    resolved = _motion.resolve_estimation_params(params)
    assert _motion.resolved_params_hash(row["resolved_params"]) == (
        _motion.resolved_params_hash(resolved)
    )
    assert _motion.resolved_params_hash(row["resolved_params"]) == (
        (MotionEstimateSelection & key).fetch1("resolved_params_hash")
    )

    n = int(DRIFT_DURATION_S * 30_000)
    assert row["n_samples"] == n
    # SpikeInterface derives the rate of a timestamped NWB series from its
    # timestamps, so it can differ from 30 kHz in the last few bits.
    assert row["sampling_frequency"] == pytest.approx(30_000.0, rel=1e-12)
    np.testing.assert_array_equal(row["continuity_spans"], [[0, n]])
    np.testing.assert_array_equal(row["statistics_spans"], [[0, n]])
    assert len(row["channel_ids"]) == 32
    np.testing.assert_array_equal(
        row["channel_locations"], computed["locations"]
    )
    diagnostics = computed["diagnostics"]
    assert row["n_peaks_detected"] == diagnostics.n_peaks_detected
    assert row["n_peaks_kept"] == diagnostics.n_peaks_kept > 0
    np.testing.assert_array_equal(
        row["peaks_per_temporal_bin"], diagnostics.peaks_per_temporal_bin
    )
    np.testing.assert_array_equal(row["noise_levels"], diagnostics.noise_levels)
    assert row["n_temporal_bins"] == motion.displacement[0].shape[0]
    # A +/-25 um drift was planted; the estimate must see it.
    assert row["max_abs_displacement_um"] > 10.0
    assert row["input_fingerprint"] == _motion.motion_input_fingerprint(
        source_content_hash=(MotionEstimateSelection & key).fetch1(
            "source_content_hash"
        ),
        artifact_detection_id=None,
        n_samples=n,
        sampling_frequency=row["sampling_frequency"],
        continuity_spans=row["continuity_spans"],
        statistics_spans=row["statistics_spans"],
        channel_ids=row["channel_ids"],
        channel_locations=row["channel_locations"],
        resolved_params_hash=_motion.resolved_params_hash(resolved),
    )


def test_masked_estimate_uses_the_pinned_artifact_mask(drift_recording):
    """An artifact-backed selection is a distinct estimate whose statistics
    spans exclude the masked period, and it protects its detection."""
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionEstimate,
        MotionEstimateSelection,
    )

    recording_key = drift_recording["recording_key"]
    artifact_key = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": recording_key["recording_id"],
            "artifact_detection_params_name": "none",
            "manual_excluded_times": np.array([[10.0, 12.0]]),
        }
    )
    RecordingArtifactDetection.populate(artifact_key, reserve_jobs=False)
    key = _select(recording_key, **artifact_key)

    assert key != _select(recording_key)
    assert MotionEstimateSelection.resolve_source(
        key
    ).artifact_detection_id == (artifact_key["artifact_detection_id"])
    MotionEstimate.populate(key, reserve_jobs=False)

    row = (MotionEstimate & key).fetch1()
    fs = 30_000
    n = int(DRIFT_DURATION_S * fs)
    np.testing.assert_array_equal(row["continuity_spans"], [[0, n]])
    spans = row["statistics_spans"]
    assert spans.shape == (2, 2)
    assert spans[0, 0] == 0 and spans[1, 1] == n
    assert spans[0, 1] <= 10 * fs and spans[1, 0] >= 12 * fs

    with pytest.raises(ValueError, match="MotionEstimateSelection"):
        (RecordingArtifactDetection & artifact_key).delete(safemode=False)
    assert RecordingArtifactDetection & artifact_key


def test_stale_selection_is_refused_at_compute(drift_recording, monkeypatch):
    import spikeinterface

    from spyglass.spikesorting.v2 import _motion
    from spyglass.spikesorting.v2.motion import MotionEstimate

    key = _select(drift_recording["recording_key"], "dredge_v1")

    monkeypatch.setattr(spikeinterface, "__version__", "0.0.0")
    with pytest.raises(ValueError, match="SpikeInterface version"):
        MotionEstimate.populate(key, reserve_jobs=False)
    monkeypatch.undo()

    resolve = _motion.resolve_estimation_params

    def _changed_resolution(params):
        resolved = resolve(params)
        resolved["estimate_motion_kwargs"]["bin_s"] = 2.0
        return resolved

    monkeypatch.setattr(
        _motion, "resolve_estimation_params", _changed_resolution
    )
    with pytest.raises(ValueError, match="resolved configuration hash"):
        MotionEstimate.populate(key, reserve_jobs=False)
    assert not (MotionEstimate & key)
