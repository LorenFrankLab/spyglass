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
    np.testing.assert_array_equal(
        row["continuity_end_s"],
        _timestamps_at(drift_recording["recording_key"], [n - 1]),
    )
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
        continuity_end_s=row["continuity_end_s"],
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


def _assert_time_map(row, spans, starts, ends):
    """The row persists the spans, their first/last timestamps and the
    recipe's clock on them."""
    from spyglass.spikesorting.v2._motion import build_estimation_clock

    np.testing.assert_array_equal(row["continuity_spans"], spans)
    np.testing.assert_array_equal(row["continuity_start_s"], starts)
    np.testing.assert_array_equal(row["continuity_end_s"], ends)
    expected = build_estimation_clock(
        spans,
        starts,
        ends,
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
    _assert_time_map(
        row, spans, starts, _timestamps_at(recording_key, spans[:, 1] - 1)
    )
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
    and member B's internal gap are both edges) with each span's real first
    and last timestamps; a rebuild reproducing different ones is refused."""
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
    expected_ends = _timestamps_at(
        discontinuous_sources["member_a"], [n_a - 1]
    ) + _timestamps_at(
        discontinuous_sources["member_b"], [b - 1 for _, b in b_spans]
    )
    np.testing.assert_array_equal(row["continuity_end_s"], expected_ends)
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
        row,
        concat["continuity_spans"],
        concat["continuity_start_s"],
        concat["continuity_end_s"],
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

    # The ground truth's time zero is the raw recording's first sample.
    _, _, truth = rigid_drift_recordings(seed=0, duration_s=DRIFT_DURATION_S)
    motion = MotionEstimate().get_motion(key)
    t0 = discontinuous_sources["t0"]
    clock = MotionEstimate().get_estimation_clock(key)
    clock = clock._replace(
        source_start_s=clock.source_start_s - t0,
        source_end_s=clock.source_end_s - t0,
    )
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


# ---- motion-corrected recordings ---------------------------------------------


def _populated_estimate(**source) -> dict:
    """The ``dredge_fast_v1`` estimate of a source, populated if missing."""
    from spyglass.spikesorting.v2.motion import (
        MotionEstimate,
        MotionEstimateSelection,
    )

    key = MotionEstimateSelection.insert_selection(
        {"motion_estimation_params_name": "dredge_fast_v1", **source}
    )
    if not (MotionEstimate & key):
        MotionEstimate.populate(key, reserve_jobs=False)
    return key


def _select_corrected(
    estimate_key, interpolation="kriging_force_extrapolate_v1"
) -> dict:
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecordingSelection,
        MotionInterpolationParameters,
    )

    MotionInterpolationParameters.insert_default()
    return MotionCorrectedRecordingSelection.insert_selection(
        {
            "motion_estimate_id": estimate_key["motion_estimate_id"],
            "motion_interpolation_params_name": interpolation,
        }
    )


def _populated_corrected(estimate_key, interpolation=None) -> dict:
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording

    key = _select_corrected(
        estimate_key, *(() if interpolation is None else (interpolation,))
    )
    if not (MotionCorrectedRecording & key):
        MotionCorrectedRecording.populate(key, reserve_jobs=False)
    return key


def _drop_corrected(key) -> None:
    """Delete a corrected recording row with its analysis file row and file.

    Leaves no orphaned ``AnalysisNwbfile`` row or file behind, so a test can
    repopulate the same selection.
    """
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording

    names = (MotionCorrectedRecording & key).fetch("analysis_file_name")
    paths = [AnalysisNwbfile.get_abs_path(name) for name in names]
    (MotionCorrectedRecording & key).delete_quick()
    for name, path in zip(names, paths):
        (AnalysisNwbfile & {"analysis_file_name": name}).delete_quick()
        Path(path).unlink(missing_ok=True)


def _no_estimation(*_args, **_kwargs):
    raise AssertionError("motion was estimated again")


def _file_hash(abs_path) -> str:
    from spyglass.spikesorting.v2._recompute import combined_hash
    from spyglass.spikesorting.v2._recording_fingerprint import (
        recording_content_fingerprint,
    )

    return combined_hash(
        recording_content_fingerprint(
            abs_path,
            electrical_series_path="acquisition/ProcessedElectricalSeries",
        )
    )


def test_correction_recipes_install_and_are_validated(dj_conn):
    import datajoint as dj
    from pydantic import ValidationError

    from spyglass.spikesorting.v2.motion import (
        MotionCorrectionParameters,
        MotionInterpolationParameters,
    )

    MotionCorrectionParameters.insert_default()
    MotionCorrectionParameters.insert_default()
    recipes = {
        row["motion_correction_params_name"]: row
        for row in MotionCorrectionParameters.fetch(as_dict=True)
    }
    for name in ("dredge_v1", "dredge_fast_v1"):
        assert recipes[name]["motion_estimation_params_name"] == name
        assert recipes[name]["motion_interpolation_params_name"] == (
            "kriging_force_extrapolate_v1"
        )
    assert MotionInterpolationParameters & {
        "motion_interpolation_params_name": "kriging_remove_channels_v1"
    }
    with pytest.raises(ValidationError, match="border_mode"):
        MotionInterpolationParameters.insert1(
            {
                "motion_interpolation_params_name": "zeros_test",
                "params": {
                    "border_mode": "force_zeros",
                    "spatial_interpolation_method": "kriging",
                    "sigma_um": 20.0,
                    "p": 2,
                    "num_closest": 3,
                },
            }
        )
    with pytest.raises(dj.errors.DataJointError, match="update1"):
        MotionCorrectionParameters.update1(
            {**recipes["dredge_v1"], "motion_estimation_params_name": "x"}
        )


def test_interpolation_only_change_reuses_the_estimate(
    discontinuous_sources, monkeypatch
):
    """Two interpolation recipes on one saved estimate are two corrected
    recordings with one ``motion_estimate_id``; populating both never
    estimates motion again. Selection is idempotent, content-addressed and
    guarded."""
    import datajoint as dj
    import spikeinterface

    from spyglass.spikesorting.v2 import _motion
    from spyglass.spikesorting.v2._selection_identity import deterministic_id
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionCorrectedRecordingSelection,
        MotionEstimate,
    )

    estimate = _populated_estimate(
        recording_id=discontinuous_sources["member_b"]["recording_id"]
    )
    extrapolate = _select_corrected(estimate)
    removal = _select_corrected(estimate, "kriging_remove_channels_v1")
    assert extrapolate != removal
    assert _select_corrected(estimate) == extrapolate
    rows = (MotionCorrectedRecordingSelection & [extrapolate, removal]).fetch(
        as_dict=True
    )
    assert {row["motion_estimate_id"] for row in rows} == {
        estimate["motion_estimate_id"]
    }
    row = (MotionCorrectedRecordingSelection & extrapolate).fetch1()
    assert row["spikeinterface_version"] == spikeinterface.__version__
    assert extrapolate["motion_corrected_recording_id"] == deterministic_id(
        "motion_corrected_recording",
        _motion.motion_corrected_identity_payload(
            motion_estimate_id=estimate["motion_estimate_id"],
            motion_interpolation_params_name="kriging_force_extrapolate_v1",
            resolved_params_hash=row["resolved_params_hash"],
            spikeinterface_version=row["spikeinterface_version"],
            motion_interpolation_algorithm_version=(
                _motion.MOTION_INTERPOLATION_ALGORITHM_VERSION
            ),
        ),
    )

    n_estimates = len(MotionEstimate())
    monkeypatch.setattr(_motion, "estimate_motion_in_spans", _no_estimation)
    MotionCorrectedRecording.populate(
        [extrapolate, removal], reserve_jobs=False
    )
    assert len(MotionCorrectedRecording & [extrapolate, removal]) == 2
    assert len(MotionEstimate()) == n_estimates

    with pytest.raises(dj.errors.DataJointError, match="insert_selection"):
        MotionCorrectedRecordingSelection.insert1(
            {**row, "motion_corrected_recording_id": uuid.uuid4()}
        )
    with pytest.raises(ValueError, match="does not match the id derived"):
        MotionCorrectedRecordingSelection.insert_selection(
            {
                "motion_estimate_id": estimate["motion_estimate_id"],
                "motion_interpolation_params_name": (
                    "kriging_force_extrapolate_v1"
                ),
                "motion_corrected_recording_id": uuid.uuid4(),
            }
        )
    unpopulated = _select(discontinuous_sources["member_b"], "dredge_v1")
    with pytest.raises(ValueError, match="not populated"):
        _select_corrected(unpopulated)


def test_stale_corrected_selection_is_refused_at_compute(
    discontinuous_sources, monkeypatch
):
    """A selection made under another SpikeInterface version, or whose
    interpolation recipe now resolves differently, is refused at compute and
    leaves no row."""
    import spikeinterface

    from spyglass.spikesorting.v2 import _motion
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionCorrectedRecordingSelection,
        MotionInterpolationParameters,
    )

    MotionInterpolationParameters.insert1(
        {
            "motion_interpolation_params_name": "stale_test_idw",
            "params": {
                "border_mode": "force_extrapolate",
                "spatial_interpolation_method": "idw",
                "sigma_um": 20.0,
                "p": 2,
                "num_closest": 3,
            },
        },
        skip_duplicates=True,
    )
    estimate = _populated_estimate(
        recording_id=discontinuous_sources["member_b"]["recording_id"]
    )
    key = _select_corrected(estimate, "stale_test_idw")
    try:
        monkeypatch.setattr(spikeinterface, "__version__", "0.0.0")
        with pytest.raises(ValueError, match="SpikeInterface version"):
            MotionCorrectedRecording.populate(key, reserve_jobs=False)
        monkeypatch.undo()

        resolve = _motion.resolve_interpolation_params

        def _changed_resolution(params):
            return {**resolve(params), "sigma_um": 30.0}

        monkeypatch.setattr(
            _motion, "resolve_interpolation_params", _changed_resolution
        )
        with pytest.raises(ValueError, match="resolved interpolation hash"):
            MotionCorrectedRecording.populate(key, reserve_jobs=False)
        assert not (MotionCorrectedRecording & key)
    finally:
        monkeypatch.undo()
        (MotionCorrectedRecordingSelection & key).super_delete(
            warn=False, safemode=False
        )


@pytest.mark.parametrize("source", ["gapped_recording", "concat"])
def test_estimate_and_corrected_recording_round_trip(
    discontinuous_sources, monkeypatch, source
):
    """The reloaded corrected traces are the in-memory corrected traces,
    written with the source artifact's own timestamps (a recording with an
    acquisition gap; a concatenation's own clock). The row carries the
    channel map, the estimate's spans and the source's content hash; the
    persisted series references those electrodes; the content hash is the
    file's fingerprint and survives a rebuild."""
    import h5py

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import _motion
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionEstimate,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    if source == "concat":
        source_key = discontinuous_sources["concat_key"]
        table = ConcatenatedRecording
        estimate = _populated_estimate(
            concat_recording_id=source_key["concat_recording_id"]
        )
    else:
        source_key = discontinuous_sources["member_b"]
        table = Recording
        estimate = _populated_estimate(recording_id=source_key["recording_id"])
    key = _select_corrected(estimate)
    _drop_corrected(key)

    captured = {}
    apply = _motion.apply_motion_on_estimation_clock

    def _capture(*args, **kwargs):
        applied = apply(*args, **kwargs)
        captured["traces"] = applied.recording.get_traces()
        return applied

    monkeypatch.setattr(_motion, "apply_motion_on_estimation_clock", _capture)
    MotionCorrectedRecording.populate(key, reserve_jobs=False)
    monkeypatch.undo()

    row = (MotionCorrectedRecording & key).fetch1()
    estimate_row = (MotionEstimate & estimate).fetch1()
    source_recording = table().get_recording(source_key)
    corrected = MotionCorrectedRecording().get_recording(key)
    n = source_recording.get_num_samples()

    reloaded = corrected.get_traces()
    np.testing.assert_array_equal(reloaded, captured["traces"])
    # Motion was planted; the correction changed the traces.
    assert np.max(np.abs(reloaded - source_recording.get_traces())) > 10.0
    np.testing.assert_array_equal(
        corrected.get_times(), source_recording.get_times()
    )
    assert row["n_samples"] == corrected.get_num_samples() == n
    assert row["sampling_frequency"] == estimate_row["sampling_frequency"]
    assert row["source_content_hash"] == (table & source_key).fetch1(
        "content_hash"
    )
    ids = source_recording.channel_ids.tolist()
    assert list(row["channel_ids"]) == corrected.channel_ids.tolist() == ids
    assert row["n_channels"] == len(ids)
    assert list(row["removed_channel_ids"]) == []
    np.testing.assert_array_equal(
        row["channel_locations"], estimate_row["channel_locations"]
    )
    np.testing.assert_array_equal(
        np.asarray(corrected.get_channel_locations())[:, :2],
        estimate_row["channel_locations"],
    )
    np.testing.assert_array_equal(
        row["statistics_spans"], estimate_row["statistics_spans"]
    )
    np.testing.assert_array_equal(
        row["continuity_spans"], estimate_row["continuity_spans"]
    )

    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    with h5py.File(abs_path, "r") as handle:
        series = handle[row["electrical_series_path"]]
        region = series["electrodes"][:]
        electrode_ids = handle["general/extracellular_ephys/electrodes/id"][:]
        assert "motion corrected" in series.attrs["filtering"]
        assert "Motion-corrected" in series.attrs["description"]
    assert electrode_ids[region].tolist() == [int(c) for c in ids]
    assert _file_hash(abs_path) == row["content_hash"]


    # Rebuilt from the saved motion (the estimator must not run): same hash,
    # same traces.
    from pathlib import Path

    Path(abs_path).unlink()
    monkeypatch.setattr(_motion, "estimate_motion_in_spans", _no_estimation)
    rebuilt = MotionCorrectedRecording().get_recording(key)
    assert _file_hash(abs_path) == row["content_hash"]
    np.testing.assert_array_equal(rebuilt.get_traces(), reloaded)


def test_remove_channels_records_the_removed_contacts(discontinuous_sources):
    """The planted +/-25 um drift moves the end contacts off the probe in
    some bin: ``remove_channels`` drops them, records them, and keeps the
    other channels in order at their unmoved positions."""
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionEstimate,
    )
    from spyglass.spikesorting.v2.recording import Recording

    recording_key = discontinuous_sources["member_b"]
    estimate = _populated_estimate(recording_id=recording_key["recording_id"])
    key = _populated_corrected(estimate, "kriging_remove_channels_v1")

    row = (MotionCorrectedRecording & key).fetch1()
    source_ids = Recording().get_recording(recording_key).channel_ids.tolist()
    removed = list(row["removed_channel_ids"])
    kept = [c for c in source_ids if c not in removed]
    assert removed and set(removed) <= set(source_ids)
    assert list(row["channel_ids"]) == kept
    assert row["n_channels"] == len(kept) == len(source_ids) - len(removed)
    corrected = MotionCorrectedRecording().get_recording(key)
    assert corrected.channel_ids.tolist() == kept
    locations = (MotionEstimate & estimate).fetch1("channel_locations")
    np.testing.assert_array_equal(
        row["channel_locations"],
        locations[[source_ids.index(c) for c in kept]],
    )


def test_masked_frames_of_the_corrected_recording_are_zero(drift_recording):
    """A masked estimate's corrected recording is zero exactly outside the
    estimate's statistics spans and nonzero inside them."""
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording

    recording_key = drift_recording["recording_key"]
    artifact_key = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": recording_key["recording_id"],
            "artifact_detection_params_name": "none",
            "manual_excluded_times": np.array([[20.0, 21.0]]),
        }
    )
    RecordingArtifactDetection.populate(artifact_key, reserve_jobs=False)
    estimate = _populated_estimate(
        recording_id=recording_key["recording_id"], **artifact_key
    )
    key = _populated_corrected(estimate)

    spans = (MotionCorrectedRecording & key).fetch1("statistics_spans")
    assert spans.shape == (2, 2)
    traces = MotionCorrectedRecording().get_recording(key).get_traces()
    masked = slice(int(spans[0, 1]), int(spans[1, 0]))
    assert masked.stop - masked.start > 25_000
    assert np.all(traces[masked] == 0)
    for start, end in spans:
        assert np.all(np.any(traces[start:end] != 0, axis=1))


def test_motion_failure_cleanup_and_cache_rebuild(
    discontinuous_sources, monkeypatch
):
    """A failed write or registration leaves no row and no file; a retry
    succeeds. A deleted artifact is rebuilt from the saved motion (the
    estimator is disabled) with the stored hash. A rebuild whose hash
    differs is refused, leaves nothing behind, and never replaces a present
    artifact."""
    import hashlib
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import _motion, _recording_nwb
    from spyglass.spikesorting.v2.exceptions import (
        RecordingContentDriftError,
    )
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording
    from spyglass.spikesorting.v2.recording import Recording

    recording_key = discontinuous_sources["member_b"]
    estimate = _populated_estimate(recording_id=recording_key["recording_id"])
    key = _select_corrected(estimate)
    _drop_corrected(key)
    source_file = (Recording & recording_key).fetch1("analysis_file_name")
    folder = Path(AnalysisNwbfile.get_abs_path(source_file)).parent
    nwb_file_name = (
        AnalysisNwbfile & {"analysis_file_name": source_file}
    ).fetch1("nwb_file_name")

    def _snapshot():
        return (
            set(folder.glob("*.nwb")),
            len(AnalysisNwbfile & {"nwb_file_name": nwb_file_name}),
        )

    before = _snapshot()

    def _fail(*_args, **_kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(_recording_nwb, "_persist_channel_geometry", _fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        MotionCorrectedRecording.populate(key, reserve_jobs=False)
    monkeypatch.undo()
    assert not (MotionCorrectedRecording & key)
    assert _snapshot() == before

    monkeypatch.setattr(AnalysisNwbfile, "add", _fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        MotionCorrectedRecording.populate(key, reserve_jobs=False)
    monkeypatch.undo()
    assert not (MotionCorrectedRecording & key)
    assert _snapshot() == before

    MotionCorrectedRecording.populate(key, reserve_jobs=False)
    row = (MotionCorrectedRecording & key).fetch1()
    abs_path = Path(AnalysisNwbfile.get_abs_path(row["analysis_file_name"]))
    traces = MotionCorrectedRecording().get_recording(key).get_traces()

    monkeypatch.setattr(_motion, "estimate_motion_in_spans", _no_estimation)
    abs_path.unlink()
    rebuilt = MotionCorrectedRecording().get_recording(key)
    assert _file_hash(str(abs_path)) == row["content_hash"]
    np.testing.assert_array_equal(rebuilt.get_traces(), traces)

    apply = _motion.apply_motion_on_estimation_clock

    def _drifted(*args, **kwargs):
        import spikeinterface.preprocessing as sip

        applied = apply(*args, **kwargs)
        return applied._replace(
            recording=sip.scale(applied.recording, gain=1.5)
        )

    monkeypatch.setattr(_motion, "apply_motion_on_estimation_clock", _drifted)
    present = hashlib.sha256(abs_path.read_bytes()).hexdigest()
    MotionCorrectedRecording()._rebuild_nwb_artifact(key)
    assert hashlib.sha256(abs_path.read_bytes()).hexdigest() == present

    abs_path.unlink()
    files = set(folder.glob("*.nwb"))
    with pytest.raises(RecordingContentDriftError, match="does not match"):
        MotionCorrectedRecording().get_recording(key)
    assert not abs_path.exists()
    assert set(folder.glob("*.nwb")) == files
    monkeypatch.undo()
    MotionCorrectedRecording().get_recording(key)
    assert _file_hash(str(abs_path)) == row["content_hash"]


def test_effective_traces_resolve_a_corrected_recording(discontinuous_sources):
    """``ensure_effective_traces`` self-heals a missing corrected artifact
    through its own table, and ``read_effective_recording`` then reads the
    corrected traces as stored, without masking them again."""
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._source_resolution import (
        EffectiveTraces,
        read_effective_recording,
    )
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording
    from spyglass.spikesorting.v2.sorting import SortingSelection

    estimate = _populated_estimate(
        recording_id=discontinuous_sources["member_b"]["recording_id"]
    )
    key = _populated_corrected(estimate)
    row = (MotionCorrectedRecording & key).fetch1()
    expected = MotionCorrectedRecording().get_recording(key).get_traces()
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    Path(abs_path).unlink()

    traces = EffectiveTraces(
        kind="motion_corrected_recording",
        key=key,
        row=row,
        apply_artifact_mask=False,
    )
    SortingSelection.ensure_effective_traces(traces)
    assert Path(abs_path).exists()
    np.testing.assert_array_equal(
        read_effective_recording(abs_path, traces).get_traces(), expected
    )
