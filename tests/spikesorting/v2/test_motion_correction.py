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
                    "max_gap_s": 30.0,
                },
            }
        )
    assert not (motion_params & {"motion_estimation_params_name": "typo_row"})


def test_estimation_row_rejects_job_kwargs_seed(motion_params):
    with pytest.raises(ValueError, match="noise_levels_seed"):
        motion_params.insert1(
            {
                "motion_estimation_params_name": "seeded_job_kwargs",
                "params": {"preset": "rigid_fast", "max_gap_s": 30.0},
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
                    preset="dredge", max_gap_s=30.0
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
    from spyglass.spikesorting.v2.recording import Recording

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
    # One span: the estimation clock is the source's own first timestamp on.
    first_timestamp = float(
        Recording()
        .get_recording(drift_recording["recording_key"])
        .sample_index_to_time(0)
    )
    np.testing.assert_array_equal(row["continuity_start_s"], [first_timestamp])
    np.testing.assert_array_equal(row["estimation_start_s"], [first_timestamp])
    np.testing.assert_array_equal(
        row["peaks_per_continuity_span"], [row["n_peaks_kept"]]
    )
    clock = MotionEstimate().get_estimation_clock(key)
    np.testing.assert_array_equal(clock.spans, row["continuity_spans"])
    assert clock.sampling_frequency == row["sampling_frequency"]
    mapped = MotionEstimate().get_displacement_on_source_clock(key)
    assert not mapped.in_gap.any()
    np.testing.assert_allclose(
        mapped.source_time_s, stored.temporal_bins_s[0], rtol=0, atol=1e-9
    )
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
        continuity_start_s=row["continuity_start_s"],
        statistics_spans=row["statistics_spans"],
        channel_ids=row["channel_ids"],
        channel_locations=row["channel_locations"],
        resolved_params_hash=_motion.resolved_params_hash(resolved),
    )


def test_masked_estimate_uses_the_pinned_artifact_mask(drift_recording):
    """An artifact-backed selection is a distinct estimate whose statistics
    spans exclude the masked period, and it protects its detection.

    The mask starts 5 frames after the deepest trough in [10.0, 10.1) s: that
    spike stays unmasked and is detected, but its localization window (3
    frames before, 9 from the peak) reaches into the mask, so it is dropped.
    """
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionEstimate,
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.recording import Recording

    recording_key = drift_recording["recording_key"]
    recording = Recording().get_recording(recording_key)
    window_start = int(10.0 * 30_000)
    troughs = recording.get_traces(
        start_frame=window_start, end_frame=window_start + 3_000
    ).min(axis=1)
    trough = window_start + int(np.argmin(troughs))
    mask_start = float(recording.sample_index_to_time(trough + 5))
    artifact_key = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": recording_key["recording_id"],
            "artifact_detection_params_name": "none",
            "manual_excluded_times": np.array([[mask_start, 12.0]]),
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
    assert trough < spans[0, 1] <= trough + 9 and spans[1, 0] >= 12 * fs
    # Peaks whose window touches the mask (or the recording's ends) are
    # dropped, and the masked estimate is not the unmasked one.
    assert row["n_peaks_kept"] < row["n_peaks_detected"]
    unmasked_key = _select(recording_key)
    if not (MotionEstimate & unmasked_key):
        MotionEstimate.populate(unmasked_key, reserve_jobs=False)
    unmasked = (MotionEstimate & unmasked_key).fetch1()
    assert unmasked["n_peaks_kept"] > row["n_peaks_kept"]
    assert not np.array_equal(
        MotionEstimate().get_motion(key).displacement[0],
        MotionEstimate().get_motion(unmasked_key).displacement[0],
    )

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


# ---- discontinuous sources ---------------------------------------------------
#
# Two more recordings of the planted-drift session: member A keeps [16, 20) s;
# member B keeps [23, 26) s and [27, end) s, so it has an acquisition gap of
# its own. B alone is the gapped single-recording source; A then B is a
# two-member concatenation with a 3 s wall-clock gap at the join. Both members
# start in the same power-of-two range of timestamps: SpikeInterface derives a
# timestamped series' rate from its first 1000 timestamp steps
# (``extractors/nwbextractors.py:369``), whose float rounding changes between
# such ranges, and concatenation requires the members' rates to agree within
# 1e-9 Hz.
MEMBER_A_INTERVAL = "motion member a"
MEMBER_B_INTERVAL = "motion member b"
CONCAT_GROUP = "motion_concat"


def _drop_concat_motion_selections(concat_key) -> None:
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    keys = (
        MotionEstimateSelection.ConcatenatedRecordingSource & concat_key
    ).fetch("KEY", as_dict=True)
    if keys:
        (MotionEstimateSelection & keys).super_delete(
            warn=False, safemode=False
        )


@pytest.fixture(scope="module")
def discontinuous_sources(drift_recording):
    """A gapped ``Recording`` and a two-member ``ConcatenatedRecording``."""
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        SessionGroup,
    )
    from tests.spikesorting.v2._concat_helpers import select_unmasked_concat
    from tests.spikesorting.v2._ingest_helpers import (
        clean_session_groups_for_owner,
        configure_v2_run_inputs,
    )

    nwb_file_name = drift_recording["nwb_file_name"]
    valid = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0, t_end = float(valid[0][0]), float(valid[-1][1])
    intervals = {
        MEMBER_A_INTERVAL: [[t0 + 16.0, t0 + 20.0]],
        MEMBER_B_INTERVAL: [[t0 + 23.0, t0 + 26.0], [t0 + 27.0, t_end]],
    }
    recording_keys = {}
    members = []
    for name, times in intervals.items():
        IntervalList.insert1(
            {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": name,
                "valid_times": np.asarray(times, dtype=float),
                "pipeline": "motion_estimate_test",
            },
            skip_duplicates=True,
        )
        run = configure_v2_run_inputs(
            nwb_file_name, MOTION_TEAM, interval_list_name=name
        )
        members.append(run)
        recording_keys[name] = RecordingSelection.insert_selection(
            {**run, "preprocessing_params_name": "default"}
        )
        _drop_motion_selections(recording_keys[name])
        if not (Recording & recording_keys[name]):
            Recording.populate(recording_keys[name], reserve_jobs=False)

    clean_session_groups_for_owner(MOTION_TEAM)
    SessionGroup.create_group(MOTION_TEAM, CONCAT_GROUP, members)
    concat_key = select_unmasked_concat(
        {
            "session_group_owner": MOTION_TEAM,
            "session_group_name": CONCAT_GROUP,
            "preprocessing_params_name": "default",
        }
    )
    ConcatenatedRecording.populate(concat_key, reserve_jobs=False)

    yield {
        "t0": t0,
        "member_a": recording_keys[MEMBER_A_INTERVAL],
        "member_b": recording_keys[MEMBER_B_INTERVAL],
        "concat_key": concat_key,
    }

    _drop_concat_motion_selections(concat_key)
    for key in recording_keys.values():
        _drop_motion_selections(key)
    clean_session_groups_for_owner(MOTION_TEAM)


def _timestamps_at(recording_key, frames):
    from spyglass.spikesorting.v2.recording import Recording

    recording = Recording().get_recording(recording_key)
    return [float(recording.sample_index_to_time(int(f))) for f in frames]


def _assert_time_map(row, spans, starts):
    """The row persists ``spans``/``starts`` and the recipe's clock on them."""
    from spyglass.spikesorting.v2._motion import build_estimation_clock

    np.testing.assert_array_equal(row["continuity_spans"], spans)
    np.testing.assert_array_equal(row["continuity_start_s"], starts)
    expected = build_estimation_clock(
        spans,
        starts,
        row["sampling_frequency"],
        row["resolved_params"]["max_gap_s"],
    )
    np.testing.assert_array_equal(
        row["estimation_start_s"], expected.estimation_start_s
    )


def test_gapped_recording_is_estimated_on_one_clock(discontinuous_sources):
    """A Recording with two disjoint selected intervals is estimated once:
    both spans kept peaks, the time map is persisted, and the bins cover the
    whole estimation clock, the capped gap included."""
    from spyglass.spikesorting.v2.motion import MotionEstimate
    from spyglass.spikesorting.v2.recording import Recording

    recording_key = discontinuous_sources["member_b"]
    recording = Recording().get_recording(recording_key)
    n = recording.get_num_samples()
    key = _select(recording_key)
    MotionEstimate.populate(key, reserve_jobs=False)

    row = (MotionEstimate & key).fetch1()
    spans = row["continuity_spans"]
    assert spans.shape == (2, 2) and spans[0, 0] == 0 and spans[1, 1] == n
    starts = _timestamps_at(recording_key, spans[:, 0])
    t0 = discontinuous_sources["t0"]
    assert starts == pytest.approx([t0 + 23.0, t0 + 27.0], abs=1e-3)
    _assert_time_map(row, spans, starts)
    # The 1 s gap is below the 30 s cap, so it keeps its real length.
    assert row["estimation_start_s"][1] - row["estimation_start_s"][0] == (
        pytest.approx(4.0, abs=1e-3)
    )
    assert (row["peaks_per_continuity_span"] > 0).all()
    np.testing.assert_array_equal(row["statistics_spans"], spans)

    motion = MotionEstimate().get_motion(key)
    assert len(motion.displacement) == 1
    bins = motion.temporal_bins_s[0]
    assert bins[0] < row["estimation_start_s"][0] + 1.0
    assert (
        bins[-1]
        > row["estimation_start_s"][1]
        + (spans[1, 1] - spans[1, 0]) / row["sampling_frequency"]
        - 1.0
    )
    mapped = MotionEstimate().get_displacement_on_source_clock(key)
    assert int(mapped.in_gap.sum()) == 1
    assert np.isnan(mapped.source_time_s[mapped.in_gap]).all()
    assert set(mapped.continuity_span[~mapped.in_gap]) == {0, 1}


def test_concat_persists_its_continuity_and_rebuild_verifies_it(
    discontinuous_sources, monkeypatch
):
    """The concat row stores one continuity span per member span (the join
    and member B's internal gap are both edges) with each span's real start
    time; a rebuild reproducing different start times is refused."""
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import _concat_recording
    from spyglass.spikesorting.v2.exceptions import (
        RecordingContentDriftError,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    concat_key = discontinuous_sources["concat_key"]
    row = (ConcatenatedRecording & concat_key).fetch1()
    n_a = (
        Recording()
        .get_recording(discontinuous_sources["member_a"])
        .get_num_samples()
    )
    b_spans = _motion_spans_of(discontinuous_sources["member_b"])
    expected_spans = [[0, n_a]] + [[n_a + a, n_a + b] for a, b in b_spans]
    np.testing.assert_array_equal(row["continuity_spans"], expected_spans)
    expected_starts = _timestamps_at(
        discontinuous_sources["member_a"], [0]
    ) + _timestamps_at(
        discontinuous_sources["member_b"], [a for a, _ in b_spans]
    )
    np.testing.assert_array_equal(row["continuity_start_s"], expected_starts)
    np.testing.assert_array_equal(row["statistics_spans"], expected_spans)

    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    Path(abs_path).unlink()
    ConcatenatedRecording().get_recording(concat_key)
    assert Path(abs_path).exists()

    real = _concat_recording.concat_continuity

    def _shifted(*args, **kwargs):
        continuity = real(*args, **kwargs)
        return continuity._replace(
            start_s=[t + 1.0 for t in continuity.start_s]
        )

    monkeypatch.setattr(_concat_recording, "concat_continuity", _shifted)
    Path(abs_path).unlink()
    with pytest.raises(RecordingContentDriftError, match="continuity_start_s"):
        ConcatenatedRecording()._rebuild_nwb_artifact(concat_key)
    monkeypatch.undo()
    ConcatenatedRecording().get_recording(concat_key)


def _motion_spans_of(recording_key):
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        boundary_spans_from_timestamps,
    )
    from spyglass.spikesorting.v2.recording import Recording

    return boundary_spans_from_timestamps(
        Recording().get_recording(recording_key)
    )


def test_concat_source_is_estimated_end_to_end(discontinuous_sources):
    """A two-member concat populates end to end: one estimate over the
    persisted continuity spans (a 3 s member join and a 1 s internal gap),
    every span contributing peaks, on one estimation clock."""
    from spyglass.spikesorting.v2.motion import (
        MotionEstimate,
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    concat_key = discontinuous_sources["concat_key"]
    key = MotionEstimateSelection.insert_selection(
        {
            "concat_recording_id": concat_key["concat_recording_id"],
            "motion_estimation_params_name": "dredge_fast_v1",
        }
    )
    lineage = MotionEstimateSelection.resolve_source(key)
    assert lineage.kind == "concatenated_recording"
    MotionEstimate.populate(key, reserve_jobs=False)

    row = (MotionEstimate & key).fetch1()
    concat = (ConcatenatedRecording & concat_key).fetch1()
    assert row["n_samples"] == concat["n_samples"]
    _assert_time_map(
        row, concat["continuity_spans"], concat["continuity_start_s"]
    )
    np.testing.assert_array_equal(
        row["statistics_spans"], concat["statistics_spans"]
    )
    assert len(row["continuity_spans"]) == 3
    assert (row["peaks_per_continuity_span"] > 0).all()
    # The 3 s join and 1 s gap keep their real length under the cap.
    np.testing.assert_allclose(
        np.diff(row["estimation_start_s"]),
        np.diff(concat["continuity_start_s"]),
        rtol=0,
        atol=1e-3,
    )
    mapped = MotionEstimate().get_displacement_on_source_clock(key)
    assert set(mapped.continuity_span[~mapped.in_gap]) == {0, 1, 2}

    # The planted drift is recovered in one frame across all three spans: the
    # common-frame error on bins holding data is under half that of a zero
    # estimate (the drift's own spread over those bins). Re-centering each
    # span on its own data would leave errors of the order of the spans'
    # different mean displacements (-10, -19 and -5 um here).
    from spikeinterface.core.motion import Motion

    from tests.spikesorting.v2._motion_fixtures import (
        common_frame_error_on_source_clock,
        rigid_drift_recordings,
    )

    _, _, truth = rigid_drift_recordings(seed=0, duration_s=DRIFT_DURATION_S)
    motion = MotionEstimate().get_motion(key)
    clock = MotionEstimate().get_estimation_clock(key)
    depths = row["channel_locations"][:, 1]
    error_rms, _ = common_frame_error_on_source_clock(
        motion, clock, truth, depths
    )
    zero = Motion(
        [np.zeros_like(motion.displacement[0])],
        motion.temporal_bins_s,
        motion.spatial_bins_um,
    )
    zero_rms, _ = common_frame_error_on_source_clock(zero, clock, truth, depths)
    assert error_rms < 0.5 * zero_rms


def test_orphan_and_bypassed_source_parts_are_refused(
    discontinuous_sources, drift_recording
):
    """``resolve_source`` refuses a master with both source parts, and a
    concat source paired with an artifact detection (raw inserts)."""
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import (
        ArtifactDetectionOutput,
    )
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    concat_id = discontinuous_sources["concat_key"]["concat_recording_id"]
    recording_id = drift_recording["recording_key"]["recording_id"]
    artifact_key = RecordingArtifactSelection.insert_selection(
        {"recording_id": recording_id, "artifact_detection_params_name": "none"}
    )
    RecordingArtifactDetection.populate(artifact_key, reserve_jobs=False)
    try:
        merge_id = ArtifactDetectionOutput.get_merge_id(artifact_key)
    except KeyError:
        ArtifactDetectionOutput.insert_detection(artifact_key)
        merge_id = ArtifactDetectionOutput.get_merge_id(artifact_key)

    def _orphan(**parts):
        row = {
            "motion_estimate_id": uuid.uuid4(),
            "motion_estimation_params_name": "dredge_v1",
            "resolved_params_hash": "0" * 64,
            "spikeinterface_version": "0.104.3",
            "motion_algorithm_version": 1,
            "source_content_hash": "0" * 64,
        }
        MotionEstimateSelection.insert1(row, allow_direct_insert=True)
        pk = {"motion_estimate_id": row["motion_estimate_id"]}
        for part, extra in parts.items():
            getattr(MotionEstimateSelection, part).insert1({**pk, **extra})
        return pk

    both = _orphan(
        RecordingSource={"recording_id": recording_id},
        ConcatenatedRecordingSource={"concat_recording_id": concat_id},
    )
    concat_masked = _orphan(
        ConcatenatedRecordingSource={"concat_recording_id": concat_id},
        ArtifactDetectionSource={"artifact_detection_merge_id": merge_id},
    )
    try:
        with pytest.raises(SchemaBypassError, match="2 source part rows"):
            MotionEstimateSelection.resolve_source(both)
        with pytest.raises(SchemaBypassError, match="pairs a concatenated"):
            MotionEstimateSelection.resolve_source(concat_masked)
    finally:
        (MotionEstimateSelection & [both, concat_masked]).super_delete(
            warn=False, safemode=False
        )
