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

from tests.spikesorting.v2._motion_db_helpers import (
    CONCAT_GROUP,
    DRIFT_DURATION_S,
    DRIFT_NWB,
    MEMBER_A_INTERVAL,
    MEMBER_B_INTERVAL,
    MOTION_TEAM,
    drop_corrected,
    drop_motion_selections,
    drop_pipeline_sorts,
    drop_sorts,
    masked_artifact,
    populated_corrected,
    populated_estimate,
    select_corrected,
    session_start_s,
    sorter_key,
)


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
        motion_estimate_selection_identity,
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
    # The planner's DB-free derivation mints the id the insert does.
    assert (
        motion_estimate_selection_identity(
            source_kind="recording",
            source_id=recording_key["recording_id"],
            source_content_hash=(Recording & recording_key).fetch1(
                "content_hash"
            ),
            artifact_detection_id=None,
            motion_estimation_params_name="dredge_fast_v1",
            estimation_params=params,
        ).selection_id
        == expected
    )
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


def test_unfiltered_sources_are_refused_for_motion_estimation(
    offset_source_concat, motion_params
):
    """A single recording and a concatenation built with the ``no_filter``
    recipe (no temporal filter) cannot be selected for motion estimation,
    with or without an artifact mask; no selection row is written."""
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    n_selections = len(MotionEstimateSelection())
    requests = [
        {"recording_id": offset_source_concat["member_a"]["recording_id"]},
        {
            "recording_id": offset_source_concat["member_a"]["recording_id"],
            **offset_source_concat["artifact_key"],
        },
        {
            "concat_recording_id": offset_source_concat["concat_key"][
                "concat_recording_id"
            ]
        },
    ]
    for request in requests:
        with pytest.raises(ValueError) as raised:
            MotionEstimateSelection.insert_selection(
                {**request, "motion_estimation_params_name": "dredge_fast_v1"}
            )
        message = str(raised.value)
        assert "'no_filter' applies no temporal filter" in message
        assert "filtered, unwhitened traces" in message
    assert len(MotionEstimateSelection()) == n_selections


def test_unfiltered_source_planted_directly_is_refused_at_compute(
    offset_source_concat, motion_params
):
    """``insert_selection`` refuses a ``no_filter`` source, but a selection
    row written straight to the table (``allow_direct_insert=True``, e.g. a
    restored backup or a bulk load) bypasses that guard. ``MotionEstimate``
    must re-check the recipe itself: ``populate`` raises the same problem
    before reading any file, and no ``MotionEstimate`` row is written."""
    from spyglass.spikesorting.v2 import _motion
    from spyglass.spikesorting.v2.motion import (
        MotionEstimate,
        MotionEstimateSelection,
        MotionEstimationParameters,
    )
    from spyglass.spikesorting.v2.recording import Recording

    recording_key = offset_source_concat["member_a"]
    params_name = "dredge_fast_v1"
    estimation_params = (
        MotionEstimationParameters
        & {"motion_estimation_params_name": params_name}
    ).fetch1("params")
    content_hash = (Recording & recording_key).fetch1("content_hash")
    motion_estimate_id, master_row = _motion.motion_estimate_selection_identity(
        source_kind="recording",
        source_id=recording_key["recording_id"],
        source_content_hash=content_hash,
        artifact_detection_id=None,
        motion_estimation_params_name=params_name,
        estimation_params=estimation_params,
    )
    key = {"motion_estimate_id": motion_estimate_id}
    MotionEstimateSelection.insert1(
        {**key, **master_row}, allow_direct_insert=True
    )
    MotionEstimateSelection.RecordingSource.insert1(
        {**key, "recording_id": recording_key["recording_id"]}
    )
    try:
        with pytest.raises(ValueError, match="no temporal filter"):
            MotionEstimate.populate(key, reserve_jobs=False)
        assert not (MotionEstimate & key)
    finally:
        (MotionEstimateSelection & key).super_delete(warn=False, safemode=False)


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


def test_artifact_part_changed_around_insert_is_refused(drift_recording):
    """``motion_estimate_id`` folds in the artifact detection, so an
    ``ArtifactDetectionSource`` part deleted from a masked selection, or
    inserted on an unmasked one, no longer gives the stored id: the source
    resolution (and so ``MotionEstimate.populate``) raises instead of
    estimating on another mask than the one selected."""
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.motion import (
        MotionEstimate,
        MotionEstimateSelection,
    )

    recording_key = drift_recording["recording_key"]
    artifact_key = masked_artifact(recording_key, [5.0, 6.0])
    masked = _select(recording_key, "dredge_v1", **artifact_key)
    unmasked = _select(recording_key, "dredge_v1")
    try:
        parts = MotionEstimateSelection.ArtifactDetectionSource
        artifact_row = (parts & masked).fetch1()
        for key, artifact_id in (
            (masked, artifact_key["artifact_detection_id"]),
            (unmasked, None),
        ):
            lineage = MotionEstimateSelection.resolve_source(key)
            assert lineage.artifact_detection_id == artifact_id

        (parts & masked).delete_quick()
        parts.insert1(
            {
                **unmasked,
                "artifact_detection_merge_id": artifact_row[
                    "artifact_detection_merge_id"
                ],
            }
        )
        for key in (masked, unmasked):
            with pytest.raises(
                SchemaBypassError, match=str(key["motion_estimate_id"])
            ):
                MotionEstimateSelection.resolve_source(key)
            with pytest.raises(SchemaBypassError, match="current parts"):
                MotionEstimate.populate(key, reserve_jobs=False)
            assert not (MotionEstimate & key)
    finally:
        (MotionEstimateSelection & [masked, unmasked]).super_delete(
            warn=False, safemode=False
        )


# ---- motion-corrected recordings ---------------------------------------------


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


def test_initialize_v2_defaults_installs_motion_recipes(dj_conn, monkeypatch):
    """The one-call default seeding installs the shipped motion recipes, and
    the default-catalog audit compares them with the shipped content."""
    from spyglass.spikesorting.v2 import (
        initialize_v2_defaults,
        verify_v2_default_catalog,
    )
    from spyglass.spikesorting.v2._pipeline_reporting import (
        _v2_default_catalog_tables,
    )
    from spyglass.spikesorting.v2.motion import MotionCorrectionParameters

    calls = []
    install = MotionCorrectionParameters.insert_default.__func__

    def _observe(cls):
        calls.append(cls)
        install(cls)

    monkeypatch.setattr(
        MotionCorrectionParameters, "insert_default", classmethod(_observe)
    )
    initialize_v2_defaults()
    assert calls == [MotionCorrectionParameters]
    for name in ("dredge_v1", "dredge_fast_v1"):
        assert MotionCorrectionParameters & {
            "motion_correction_params_name": name
        }
    audited = {table.__name__ for table, _ in _v2_default_catalog_tables()}
    assert {
        "MotionEstimationParameters",
        "MotionInterpolationParameters",
        "MotionCorrectionParameters",
    } <= audited
    assert not [
        entry
        for entry in verify_v2_default_catalog()
        if entry["table"].startswith("Motion")
    ]


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
        MotionInterpolationParameters,
    )

    estimate = populated_estimate(
        recording_id=discontinuous_sources["member_b"]["recording_id"]
    )
    extrapolate = select_corrected(estimate)
    removal = select_corrected(estimate, "kriging_remove_channels_v1")
    assert extrapolate != removal
    assert select_corrected(estimate) == extrapolate
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
    interpolation_params = (
        MotionInterpolationParameters
        & {"motion_interpolation_params_name": "kriging_force_extrapolate_v1"}
    ).fetch1("params")
    assert _motion.motion_corrected_selection_identity(
        motion_estimate_id=estimate["motion_estimate_id"],
        motion_interpolation_params_name="kriging_force_extrapolate_v1",
        interpolation_params=interpolation_params,
    ) == (
        extrapolate["motion_corrected_recording_id"],
        {
            k: row[k]
            for k in (
                "motion_estimate_id",
                "motion_interpolation_params_name",
                "resolved_params_hash",
                "spikeinterface_version",
                "motion_interpolation_algorithm_version",
            )
        },
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
        select_corrected(unpopulated)


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
    estimate = populated_estimate(
        recording_id=discontinuous_sources["member_b"]["recording_id"]
    )
    key = select_corrected(estimate, "stale_test_idw")
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
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
    )

    if source == "concat":
        source_key = discontinuous_sources["concat_key"]
        table = ConcatenatedRecording
        estimate = populated_estimate(
            concat_recording_id=source_key["concat_recording_id"]
        )
    else:
        source_key = discontinuous_sources["member_b"]
        table = Recording
        estimate = populated_estimate(recording_id=source_key["recording_id"])
    key = select_corrected(estimate)
    drop_corrected(key)

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

    import spikeinterface

    from spyglass.spikesorting.v2._nwb_provenance import (
        MOTION_CORRECTION_PROVENANCE,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecordingSelection,
        MotionInterpolationParameters,
    )

    selection = (MotionCorrectedRecordingSelection & key).fetch1()
    provenance = read_provenance_values(abs_path, MOTION_CORRECTION_PROVENANCE)
    assert provenance["spikeinterface_version"] == spikeinterface.__version__
    assert provenance["interpolation"] == _motion.resolve_interpolation_params(
        (
            MotionInterpolationParameters
            & {
                "motion_interpolation_params_name": "kriging_force_extrapolate_v1"
            }
        ).fetch1("params")
    )
    assert provenance["motion_estimate_id"] == str(
        estimate["motion_estimate_id"]
    )
    assert provenance["motion_corrected_recording_id"] == str(
        key["motion_corrected_recording_id"]
    )
    assert provenance["motion_interpolation_params_name"] == (
        selection["motion_interpolation_params_name"]
    )
    assert provenance["motion_interpolation_algorithm_version"] == (
        selection["motion_interpolation_algorithm_version"]
    )
    assert provenance["source_content_hash"] == row["source_content_hash"]
    assert provenance["removed_channel_ids"] == []

    # The file alone describes its spans, its time map and, for a concat, its
    # members: the statistics spans, each continuity span with its real start
    # and end times and its start on the estimation clock, and the concat's
    # own member back-map.
    lineage = MotionEstimateSelection.resolve_source(estimate)
    assert (
        provenance["source_kind"]
        == lineage.kind
        == ("concatenated_recording" if source == "concat" else "recording")
    )
    assert provenance["source_key"] == {
        name: str(value) for name, value in lineage.key.items()
    }
    assert provenance["statistics_spans"] == (
        estimate_row["statistics_spans"].tolist()
    )
    assert provenance["estimation_clock_sampling_frequency"] == (
        estimate_row["sampling_frequency"]
    )
    from spyglass.spikesorting.v2._nwb_provenance import (
        CONCAT_MEMBERS,
        MOTION_CONTINUITY_SPANS,
        read_long_provenance,
    )

    spans = read_long_provenance(abs_path, MOTION_CONTINUITY_SPANS)
    assert [s["span_index"] for s in spans] == list(range(len(spans)))
    np.testing.assert_array_equal(
        [[s["start_sample"], s["end_sample"]] for s in spans],
        estimate_row["continuity_spans"],
    )
    for field, column in (
        ("source_start_s", "continuity_start_s"),
        ("source_end_s", "continuity_end_s"),
        ("estimation_start_s", "estimation_start_s"),
    ):
        np.testing.assert_array_equal(
            [s[field] for s in spans], estimate_row[column]
        )
    if source == "concat":
        concat_path = AnalysisNwbfile.get_abs_path(
            (ConcatenatedRecording & source_key).fetch1("analysis_file_name")
        )
        members = read_long_provenance(abs_path, CONCAT_MEMBERS)
        assert members == read_long_provenance(concat_path, CONCAT_MEMBERS)
        boundaries = (ConcatenatedRecording.MemberBoundary & source_key).fetch(
            "member_index", "end_sample", order_by="member_index"
        )
        snapshots = (
            ConcatenatedRecordingSelection.MemberSnapshot & source_key
        ).fetch("recording_id", order_by="member_index")
        assert [m["member_index"] for m in members] == list(boundaries[0])
        assert [m["concat_end_sample"] for m in members] == list(boundaries[1])
        assert [m["recording_id"] for m in members] == [
            str(r) for r in snapshots
        ]
    else:
        with pytest.raises(KeyError):
            read_long_provenance(abs_path, CONCAT_MEMBERS)

    # Rebuilt from the saved motion (the estimator must not run): same hash,
    # same traces.
    from pathlib import Path

    Path(abs_path).unlink()
    monkeypatch.setattr(_motion, "estimate_motion_in_spans", _no_estimation)
    rebuilt = MotionCorrectedRecording().get_recording(key)
    assert _file_hash(abs_path) == row["content_hash"]
    np.testing.assert_array_equal(rebuilt.get_traces(), reloaded)


def test_persisted_corrected_traces_match_the_interpolation_oracle(
    discontinuous_sources,
):
    """On a gapped recording whose gap exceeds the recipe's cap (so its
    estimation clock differs from its acquisition clock after the gap), the
    persisted corrected traces equal SpikeInterface's ``interpolate_motion``
    applied, with the recipe's explicit arguments and the saved estimate, to
    the masked source carrying the saved estimation clock as its time vector.
    Interpolating on the acquisition clock gives different traces."""
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    from spyglass.spikesorting.v2._motion import (
        estimation_times,
        resolve_interpolation_params,
    )
    from spyglass.spikesorting.v2._params.motion_estimation import (
        MotionEstimationParamsSchema,
    )
    from spyglass.spikesorting.v2._recording_geometry import (
        flatten_planar_geometry,
    )
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        complement_frame_ranges,
        silence_frame_ranges,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionEstimate,
        MotionEstimateSelection,
        MotionEstimationParameters,
        MotionInterpolationParameters,
    )
    from spyglass.spikesorting.v2.recording import Recording

    # Member B's 1 s acquisition gap, capped to 0.25 s on the clock.
    MotionEstimationParameters.insert1(
        {
            "motion_estimation_params_name": "dredge_fast_gap_cap_test",
            "params": MotionEstimationParamsSchema(
                preset="dredge_fast", max_gap_s=0.25
            ).model_dump(),
        },
        skip_duplicates=True,
    )
    recording_key = discontinuous_sources["member_b"]
    estimate = MotionEstimateSelection.insert_selection(
        {
            "recording_id": recording_key["recording_id"],
            "motion_estimation_params_name": "dredge_fast_gap_cap_test",
        }
    )
    if not (MotionEstimate & estimate):
        MotionEstimate.populate(estimate, reserve_jobs=False)
    key = populated_corrected(estimate)

    clock = MotionEstimate().get_estimation_clock(estimate)
    assert len(clock.spans) == 2
    assert clock.source_start_s[1] - clock.estimation_start_s[1] == (
        pytest.approx(0.75, abs=1e-3)
    )
    statistics = (MotionEstimate & estimate).fetch1("statistics_spans")
    motion = MotionEstimate().get_motion(estimate)
    kwargs = resolve_interpolation_params(
        (
            MotionInterpolationParameters
            & {
                "motion_interpolation_params_name": "kriging_force_extrapolate_v1"
            }
        ).fetch1("params")
    )

    def _interpolated(times):
        source = Recording().get_recording(recording_key)
        flatten_planar_geometry(source)
        n = source.get_num_samples()
        if times is not None:
            source.set_times(times(n), with_warning=False)
        masked = silence_frame_ranges(
            source,
            complement_frame_ranges([tuple(s) for s in statistics], n),
        )
        return interpolate_motion(masked, motion, **kwargs).get_traces()

    expected = _interpolated(lambda n: estimation_times(clock, np.arange(n)))
    persisted = MotionCorrectedRecording().get_recording(key)
    np.testing.assert_array_equal(persisted.get_traces(), expected)
    np.testing.assert_array_equal(
        persisted.get_times(),
        Recording().get_recording(recording_key).get_times(),
    )
    assert not np.array_equal(_interpolated(None), expected)


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
    estimate = populated_estimate(recording_id=recording_key["recording_id"])
    key = populated_corrected(estimate, "kriging_remove_channels_v1")

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


def test_report_reads_the_stored_estimate_and_corrected_recording(
    discontinuous_sources,
):
    """``MotionEstimate.report`` on a gapped recording (corrected with
    ``remove_channels``, with a trace window) and on a concatenation
    (``force_extrapolate``): the summary's evidence counts, displacement and
    spans without evidence are the stored row's; the removed channels are
    the corrected row's and equal the border contacts SpikeInterface's
    criterion names; the concatenation's member join is told apart from
    member B's own acquisition gap; the trace panel plots the corrected
    recording's traces, which differ from the original's. A corrected
    recording of another estimate, or a trace window without one, is
    refused."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spyglass.spikesorting.v2._motion_report import frame_of_source_time
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionEstimate,
    )
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    def _assert_stored(summary, estimate, corrected_key):
        row = (MotionEstimate & estimate).fetch1()
        assert [s["n_peaks_kept"] for s in summary["continuity_spans"]] == (
            row["peaks_per_continuity_span"].tolist()
        )
        assert summary["spans_without_evidence"] == (
            MotionEstimate().get_spans_without_evidence(estimate)
        )
        assert summary["max_abs_displacement_um"] == (
            row["max_abs_displacement_um"]
        )
        assert summary["n_temporal_bins"] == row["n_temporal_bins"]
        assert summary["motion_estimate_id"] == estimate["motion_estimate_id"]
        corrected = (MotionCorrectedRecording & corrected_key).fetch1()
        assert summary["removed_channel_ids"] == list(
            corrected["removed_channel_ids"]
        )
        return row

    try:
        gapped = populated_estimate(
            recording_id=discontinuous_sources["member_b"]["recording_id"]
        )
        removed_key = populated_corrected(gapped, "kriging_remove_channels_v1")
        clock = MotionEstimate().get_estimation_clock(gapped)
        window = (clock.source_start_s[0] + 1.0, clock.source_start_s[0] + 1.1)
        kept = list(
            (MotionCorrectedRecording & removed_key).fetch1("channel_ids")
        )
        channel = kept[len(kept) // 2]
        fig, summary = MotionEstimate().report(
            gapped["motion_estimate_id"],
            removed_key["motion_corrected_recording_id"],
            trace_window_s=window,
            trace_channel_ids=[channel],
        )
        _assert_stored(summary, gapped, removed_key)
        assert summary["border_mode"] == "remove_channels"
        assert summary["removed_channel_ids"]
        assert summary["border_channel_ids"] == summary["removed_channel_ids"]
        assert summary["extrapolated_channel_ids"] == []
        [gap] = summary["gaps"]
        assert gap["kind"] == "acquisition_gap"
        assert gap["source_gap_s"] == pytest.approx(1.0, abs=1e-3)

        axes = {ax.get_label(): ax for ax in fig.axes}
        traces = axes["traces"]
        assert [t.get_text() for t in traces.get_yticklabels()] == [
            str(channel)
        ]
        start, end = (int(f) for f in frame_of_source_time(clock, window))
        expected = (
            MotionCorrectedRecording()
            .get_recording(removed_key)
            .get_traces(start_frame=start, end_frame=end, channel_ids=[channel])
        )
        original_line, corrected_line = traces.get_lines()[:2]
        np.testing.assert_array_equal(
            corrected_line.get_ydata(), expected[:, 0]
        )
        assert (
            np.max(
                np.abs(original_line.get_ydata() - corrected_line.get_ydata())
            )
            > 1.0
        )

        concat_key = discontinuous_sources["concat_key"]
        concat = populated_estimate(
            concat_recording_id=concat_key["concat_recording_id"]
        )
        extrapolated_key = populated_corrected(concat)
        _, summary = MotionEstimate().report(
            {**concat, "warnings": []}, extrapolated_key
        )
        _assert_stored(summary, concat, extrapolated_key)
        assert summary["border_mode"] == "force_extrapolate"
        assert summary["removed_channel_ids"] == []
        assert summary["extrapolated_channel_ids"] == (
            summary["border_channel_ids"]
        )
        member_a_end = (
            ConcatenatedRecording.MemberBoundary
            & concat_key
            & {"member_index": 0}
        ).fetch1("end_sample")
        joins = [g for g in summary["gaps"] if g["kind"] == "member_join"]
        assert [g["frame"] for g in joins] == [member_a_end]
        assert summary["n_acquisition_gaps"] == 1

        with pytest.raises(ValueError, match="was made from motion estimate"):
            MotionEstimate().report(concat, removed_key)
        with pytest.raises(ValueError, match="pass corrected_key too"):
            MotionEstimate().report(concat, trace_window_s=window)
    finally:
        plt.close("all")


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
    estimate = populated_estimate(
        recording_id=recording_key["recording_id"], **artifact_key
    )
    key = populated_corrected(estimate)

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
    estimate = populated_estimate(recording_id=recording_key["recording_id"])
    key = select_corrected(estimate)
    drop_corrected(key)
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


def test_corrected_rebuild_is_guarded_by_content_not_spikeinterface_version(
    discontinuous_sources, monkeypatch
):
    """Under another SpikeInterface version a new corrected compute is
    refused as stale, but a missing artifact is still rebuilt from the saved
    motion when it reproduces the stored content hash; a rebuild that does
    not is still refused and leaves nothing installed."""
    from pathlib import Path

    import spikeinterface

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import _motion
    from spyglass.spikesorting.v2.exceptions import (
        RecordingContentDriftError,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionCorrectedRecordingSelection,
    )

    estimate = populated_estimate(
        recording_id=discontinuous_sources["member_b"]["recording_id"]
    )
    key = populated_corrected(estimate)
    row = (MotionCorrectedRecording & key).fetch1()
    selected_version = (MotionCorrectedRecordingSelection & key).fetch1(
        "spikeinterface_version"
    )
    abs_path = Path(AnalysisNwbfile.get_abs_path(row["analysis_file_name"]))
    traces = MotionCorrectedRecording().get_recording(key).get_traces()

    monkeypatch.setattr(spikeinterface, "__version__", "0.0.0+other")
    assert spikeinterface.__version__ != selected_version
    table = MotionCorrectedRecording()
    with pytest.raises(ValueError, match="SpikeInterface version"):
        table.make_compute(key, *table.make_fetch(key))

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
    abs_path.unlink()
    try:
        with pytest.raises(RecordingContentDriftError, match="does not match"):
            MotionCorrectedRecording().get_recording(key)
        assert not abs_path.exists()
    finally:
        monkeypatch.undo()
        MotionCorrectedRecording().get_recording(key)
    assert _file_hash(str(abs_path)) == row["content_hash"]


@pytest.mark.parametrize("change", ["resolution", "algorithm_version"])
def test_missing_file_of_a_stale_corrected_selection_names_the_repair(
    discontinuous_sources, monkeypatch, change
):
    """A missing corrected artifact whose selection went stale (the
    interpolation recipe resolves differently, or the application algorithm
    changed) cannot be rebuilt: the error says the file is missing, why it
    cannot be rebuilt, and that everything made from it must be repopulated
    from a new selection; nothing is installed."""
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import _motion
    from spyglass.spikesorting.v2.exceptions import (
        RecordingContentDriftError,
    )
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording

    estimate = populated_estimate(
        recording_id=discontinuous_sources["member_b"]["recording_id"]
    )
    key = populated_corrected(estimate)
    row = (MotionCorrectedRecording & key).fetch1()
    abs_path = Path(AnalysisNwbfile.get_abs_path(row["analysis_file_name"]))
    MotionCorrectedRecording().get_recording(key)
    assert abs_path.exists()

    if change == "resolution":
        resolve = _motion.resolve_interpolation_params
        monkeypatch.setattr(
            _motion,
            "resolve_interpolation_params",
            lambda params: {**resolve(params), "sigma_um": 30.0},
        )
        reason = "resolved interpolation hash"
    else:
        monkeypatch.setattr(
            _motion,
            "MOTION_INTERPOLATION_ALGORITHM_VERSION",
            _motion.MOTION_INTERPOLATION_ALGORITHM_VERSION + 1,
        )
        reason = "motion interpolation algorithm version"
    abs_path.unlink()
    try:
        with pytest.raises(RecordingContentDriftError) as excinfo:
            MotionCorrectedRecording().get_recording(key)
        message = str(excinfo.value)
        for expected in (
            row["analysis_file_name"],
            "missing",
            reason,
            "cannot be rebuilt",
            "new MotionCorrectedRecordingSelection",
            "repopulate",
        ):
            assert expected in message
        assert not abs_path.exists()
    finally:
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

    estimate = populated_estimate(
        recording_id=discontinuous_sources["member_b"]["recording_id"]
    )
    key = populated_corrected(estimate)
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


# ---- sorts of motion-corrected recordings ------------------------------------


def test_off_and_estimate_preserve_sort_input(drift_recording):
    """Saving a motion estimate of a sort's source (under the sort's own
    mask) changes neither the sort's id nor the traces its sorter reads."""
    from spyglass.spikesorting.v2._selection_plan import (
        build_sorting_selection_plan,
    )
    from spyglass.spikesorting.v2.motion import MotionEstimate
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    recording_key = drift_recording["recording_key"]
    artifact_key = masked_artifact(recording_key, [5.0, 6.0])
    request = {
        "recording_id": recording_key["recording_id"],
        **artifact_key,
        **sorter_key(),
    }
    sort_key = SortingSelection.insert_selection(request)
    try:
        assert sort_key["sorting_id"] == (
            build_sorting_selection_plan(request).sorting_id
        )

        def sorter_input():
            fetched = Sorting().make_fetch(sort_key)
            assert fetched.traces.kind == "recording"
            assert fetched.motion_correction_provenance is None
            assert fetched.source_n_samples is None
            return SortingSelection.load_stored_traces(
                fetched.traces
            ).get_traces()

        before = sorter_input()
        np.testing.assert_array_equal(
            before, Recording().get_recording(recording_key).get_traces()
        )
        estimate = populated_estimate(
            recording_id=recording_key["recording_id"], **artifact_key
        )
        assert MotionEstimate & estimate
        assert SortingSelection.insert_selection(request) == sort_key
        assert SortingSelection.resolve_motion_correction(sort_key) is None
        np.testing.assert_array_equal(sorter_input(), before)
    finally:
        drop_sorts([sort_key])


def test_correction_identity_pins_source_masks_and_recipe(
    discontinuous_sources,
):
    """A sort may read only a corrected recording of its own source under
    its own mask; each corrected recording is its own idempotent sort
    identity, distinct from the uncorrected sort; a part row inserted
    around that check is refused before sorting."""
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    source_key = discontinuous_sources["member_b"]
    other_key = discontinuous_sources["member_a"]
    concat_key = discontinuous_sources["concat_key"]
    sorter = sorter_key()
    estimate = populated_estimate(recording_id=source_key["recording_id"])
    extrapolated = populated_corrected(estimate)
    removed = populated_corrected(estimate, "kriging_remove_channels_v1")
    concat_corrected = populated_corrected(
        populated_estimate(
            concat_recording_id=concat_key["concat_recording_id"]
        )
    )
    t0 = discontinuous_sources["t0"]
    artifact_key = masked_artifact(source_key, [t0 + 24.0, t0 + 24.5])
    base = {"recording_id": source_key["recording_id"], **sorter}

    sort_keys = []
    try:
        uncorrected = SortingSelection.insert_selection(base)
        sort_keys.append(uncorrected)
        first = SortingSelection.insert_selection({**base, **extrapolated})
        sort_keys.append(first)
        second = SortingSelection.insert_selection({**base, **removed})
        sort_keys.append(second)
        assert len({str(k["sorting_id"]) for k in sort_keys}) == 3
        n_selections = len(SortingSelection())
        assert SortingSelection.insert_selection({**base, **extrapolated}) == (
            first
        )
        assert len(SortingSelection()) == n_selections
        assert SortingSelection.resolve_motion_correction(first) == (
            extrapolated["motion_corrected_recording_id"]
        )
        effective = SortingSelection.resolve_effective_source(first)
        assert effective.lineage.kind == "recording"
        assert effective.lineage.key == {
            "recording_id": source_key["recording_id"]
        }
        assert effective.traces.kind == "motion_corrected_recording"
        assert effective.traces.key == extrapolated
        assert effective.traces.apply_artifact_mask is False

        concat_sort = SortingSelection.insert_selection(
            {**concat_key, **sorter, **concat_corrected}
        )
        sort_keys.append(concat_sort)
        concat_effective = SortingSelection.resolve_effective_source(
            concat_sort
        )
        assert concat_effective.lineage.kind == "concatenated_recording"
        assert concat_effective.traces.key == concat_corrected

        rejected = [
            # corrected recording of another recording
            {"recording_id": other_key["recording_id"], **extrapolated},
            # of a concatenation, for a recording sort
            {**base, **concat_corrected},
            # of a recording, for a concatenation sort
            {**concat_key, **extrapolated},
            # estimated without the sort's mask
            {**base, **artifact_key, **extrapolated},
        ]
        for request in rejected:
            with pytest.raises(ValueError, match="not made from this sort"):
                SortingSelection.insert_selection({**sorter, **request})
        assert len(SortingSelection()) == n_selections + 1

        SortingSelection.MotionCorrectionSource.insert1(
            {**uncorrected, **concat_corrected}
        )
        with pytest.raises(SchemaBypassError, match="not made from its"):
            Sorting().make_fetch(uncorrected)
        assert not (Sorting & uncorrected)
    finally:
        drop_sorts(sort_keys)


def test_sorter_correcting_motion_itself_is_rejected_for_a_corrected_source(
    discontinuous_sources,
):
    """Spykingcircus2's shipped row corrects motion by default: selecting it
    on a corrected recording is refused and nothing is inserted."""
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    SorterParameters.insert_default()
    sc2 = {"sorter": "spykingcircus2", "sorter_params_name": "default"}
    assert "apply_motion_correction" not in (SorterParameters & sc2).fetch1(
        "params"
    )
    source_key = discontinuous_sources["member_b"]
    corrected = populated_corrected(
        populated_estimate(recording_id=source_key["recording_id"])
    )
    n_selections = len(SortingSelection())
    with pytest.raises(ValueError, match="apply_motion_correction=False"):
        SortingSelection.insert_selection(
            {"recording_id": source_key["recording_id"], **sc2, **corrected}
        )
    assert len(SortingSelection()) == n_selections


def test_corrected_recording_referenced_by_a_sort_is_protected(
    discontinuous_sources,
):
    """A corrected recording a sort selected cannot be deleted from under
    it: a quick delete fails on the foreign key and a cascade that would
    leave the sort without its correction part is refused; a master left
    with only a correction part is an orphan the prune removes."""
    import datajoint as dj

    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source_key = discontinuous_sources["member_b"]
    sorter = sorter_key()
    corrected = populated_corrected(
        populated_estimate(recording_id=source_key["recording_id"])
    )
    sort_key = SortingSelection.insert_selection(
        {"recording_id": source_key["recording_id"], **sorter, **corrected}
    )
    orphan = {"sorting_id": uuid.uuid4()}
    try:
        with pytest.raises(dj.errors.IntegrityError):
            (MotionCorrectedRecording & corrected).delete_quick()
        with pytest.raises(dj.errors.DataJointError, match="master"):
            (MotionCorrectedRecording & corrected).super_delete(
                warn=False, safemode=False
            )
        assert MotionCorrectedRecording & corrected
        assert SortingSelection.resolve_motion_correction(sort_key) == (
            corrected["motion_corrected_recording_id"]
        )
        assert sort_key not in SortingSelection.prune_orphaned_selections()

        SortingSelection().insert1(
            {**orphan, **sorter}, allow_direct_insert=True
        )
        SortingSelection.MotionCorrectionSource.insert1({**orphan, **corrected})
        assert orphan in SortingSelection.prune_orphaned_selections()
        SortingSelection.prune_orphaned_selections(dry_run=False)
        assert not (SortingSelection & orphan)
        assert not (SortingSelection.MotionCorrectionSource & orphan)
        assert SortingSelection & sort_key
    finally:
        drop_sorts([sort_key, orphan])


def test_orphaned_motion_estimate_selections_are_pruned(
    discontinuous_sources,
):
    """A motion-estimate master left without a source part (alone, or with
    only its artifact part) is an orphan the prune lists and, when asked,
    deletes; a selection with its source is kept."""
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    estimate = populated_estimate(
        recording_id=discontinuous_sources["member_b"]["recording_id"]
    )
    template = (MotionEstimateSelection & estimate).fetch1()
    bare = {"motion_estimate_id": uuid.uuid4()}
    masked = {"motion_estimate_id": uuid.uuid4()}
    artifact_merge_id = uuid.uuid4()
    try:
        assert (
            estimate not in MotionEstimateSelection.prune_orphaned_selections()
        )
        for orphan in (bare, masked):
            MotionEstimateSelection().insert1(
                {**template, **orphan}, allow_direct_insert=True
            )
        # The artifact part's merge row need not exist for the orphan check.
        conn = MotionEstimateSelection.connection
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            MotionEstimateSelection.ArtifactDetectionSource.insert1(
                {**masked, "artifact_detection_merge_id": artifact_merge_id}
            )
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")

        found = MotionEstimateSelection.prune_orphaned_selections()
        assert bare in found and masked in found
        assert estimate not in found
        assert MotionEstimateSelection & bare

        MotionEstimateSelection.prune_orphaned_selections(dry_run=False)
        assert not (MotionEstimateSelection & [bare, masked])
        assert not (MotionEstimateSelection.ArtifactDetectionSource & masked)
        assert MotionEstimateSelection & estimate
    finally:
        conn = MotionEstimateSelection.connection
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (
                MotionEstimateSelection.ArtifactDetectionSource & masked
            ).delete_quick()
            (MotionEstimateSelection & [bare, masked]).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")


def test_corrected_sort_reads_the_corrected_traces(
    drift_recording, monkeypatch
):
    """A masked, corrected sort hands its sorter the corrected recording's
    persisted traces (not the source's), persists the corrected recording's
    statistics spans, and records the correction in its units NWB."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._nwb_provenance import (
        SORTING_PROVENANCE,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionCorrectedRecordingSelection,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    recording_key = drift_recording["recording_key"]
    artifact_key = masked_artifact(recording_key, [20.0, 21.0])
    estimate = populated_estimate(
        recording_id=recording_key["recording_id"], **artifact_key
    )
    corrected = populated_corrected(estimate)
    corrected_row = (MotionCorrectedRecording & corrected).fetch1()
    corrected_traces = (
        MotionCorrectedRecording().get_recording(corrected).get_traces()
    )
    spans = [(int(a), int(b)) for a, b in corrected_row["statistics_spans"]]
    assert len(spans) == 2
    source_traces = Recording().get_recording(recording_key).get_traces()
    inside = slice(*spans[0])
    assert (
        np.max(np.abs(corrected_traces[inside] - source_traces[inside])) > 10.0
    )

    sort_key = SortingSelection.insert_selection(
        {
            "recording_id": recording_key["recording_id"],
            **artifact_key,
            **sorter_key(),
            **corrected,
        }
    )
    captured = {}
    run_sorter = Sorting._run_sorter

    def _observe(*args, **kwargs):
        captured["traces"] = kwargs["recording"].get_traces()
        captured["spans"] = kwargs["statistics_spans"]
        return run_sorter(*args, **kwargs)

    try:
        monkeypatch.setattr(Sorting, "_run_sorter", staticmethod(_observe))
        Sorting.populate(sort_key, reserve_jobs=False)
        monkeypatch.undo()

        np.testing.assert_array_equal(captured["traces"], corrected_traces)
        assert list(captured["spans"]) == spans
        assert Sorting().get_statistics_spans(sort_key) == spans
        assert (Sorting & sort_key).fetch1("n_units") > 0

        abs_path = AnalysisNwbfile.get_abs_path(
            (Sorting & sort_key).fetch1("analysis_file_name")
        )
        provenance = read_provenance_values(abs_path, SORTING_PROVENANCE)
        selection = (MotionCorrectedRecordingSelection & corrected).fetch1()
        assert provenance["motion_corrected_recording_id"] == str(
            corrected["motion_corrected_recording_id"]
        )
        assert provenance["motion_estimate_id"] == str(
            estimate["motion_estimate_id"]
        )
        assert provenance["motion_estimation_params_name"] == "dredge_fast_v1"
        assert provenance["motion_interpolation_params_name"] == (
            selection["motion_interpolation_params_name"]
        )
        assert provenance["artifact_detection_id"] == str(
            artifact_key["artifact_detection_id"]
        )
        assert provenance["recording_id"] == str(recording_key["recording_id"])
    finally:
        monkeypatch.undo()
        drop_sorts([sort_key])


# ---- pipeline motion modes ----------------------------------------------------

#: A fast catalog preset (peak detection only, no internal motion correction)
#: whose preprocessing row is the drift fixture's.
PIPELINE_PRESET = "franklab_clusterless_2026_06"
MOTION_RECIPE = "dredge_fast_v1"


def _pipeline_inputs(drift_recording) -> dict:
    from tests.spikesorting.v2._ingest_helpers import configure_v2_run_inputs

    return {
        **configure_v2_run_inputs(
            drift_recording["nwb_file_name"], MOTION_TEAM
        ),
        "pipeline_preset": PIPELINE_PRESET,
    }


def _row_counts() -> dict:
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionCorrectedRecordingSelection,
        MotionEstimate,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    return {
        table.__name__: len(table())
        for table in (
            MotionEstimate,
            MotionCorrectedRecordingSelection,
            MotionCorrectedRecording,
            SortingSelection,
            Sorting,
        )
    }


def test_pipeline_motion_modes_on_one_recording(drift_recording):
    """``estimate`` saves the source's estimate and sorts exactly the ``off``
    sort; ``apply`` sorts the corrected recording of that estimate; every
    receipt states the mode, recipe, estimate, preset, corrected recording
    and removed channels, and a re-run reuses everything."""
    from spyglass.spikesorting.v2 import _motion
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionEstimate,
        MotionEstimateSelection,
        MotionEstimationParameters,
    )
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    inputs = _pipeline_inputs(drift_recording)
    motion = {"motion_correction_params_name": MOTION_RECIPE}
    sorting_ids = []
    try:
        off = run_v2_pipeline(**inputs)
        sorting_ids.append(off["sorting_id"])
        assert off["motion_mode"] == "off"
        for field in (
            "motion_correction_params_name",
            "motion_estimate_id",
            "motion_corrected_recording_id",
            "motion_estimation_preset",
            "motion_removed_channel_ids",
            "motion_spans_without_evidence",
        ):
            assert off[field] is None
        assert not {"motion_estimate", "motion_corrected_recording"} & set(
            off["stage_seconds"]
        )
        assert off["scientific_config"]["motion"]["mode"] == "off"

        estimate = run_v2_pipeline(**inputs, motion_mode="estimate", **motion)
        # Same sort as off: the id, and the stage is a reuse.
        assert estimate["sorting_id"] == off["sorting_id"]
        assert estimate["sorting_status"] == "reused"
        assert (
            SortingSelection.resolve_motion_correction(
                {"sorting_id": estimate["sorting_id"]}
            )
            is None
        )
        recording_key = {"recording_id": off["recording_id"]}
        artifact_id = off["artifact_detection_id"]
        expected_estimate = _motion.motion_estimate_selection_identity(
            source_kind="recording",
            source_id=off["recording_id"],
            source_content_hash=(Recording & recording_key).fetch1(
                "content_hash"
            ),
            artifact_detection_id=artifact_id,
            motion_estimation_params_name=MOTION_RECIPE,
            estimation_params=(
                MotionEstimationParameters
                & {"motion_estimation_params_name": MOTION_RECIPE}
            ).fetch1("params"),
        ).selection_id
        assert estimate["motion_estimate_id"] == expected_estimate
        estimate_key = {"motion_estimate_id": expected_estimate}
        assert MotionEstimate & estimate_key
        lineage = MotionEstimateSelection.resolve_source(estimate_key)
        assert lineage.key == recording_key
        assert lineage.artifact_detection_id == artifact_id
        assert estimate["motion_mode"] == "estimate"
        assert estimate["motion_correction_params_name"] == MOTION_RECIPE
        assert estimate["motion_estimation_preset"] == "dredge_fast"
        assert estimate["motion_corrected_recording_id"] is None
        assert estimate["motion_removed_channel_ids"] is None
        assert estimate["motion_spans_without_evidence"] == []
        assert estimate["motion_estimate_status"] == "computed"
        assert estimate["scientific_config"]["motion"]["mode"] == "estimate"
        assert estimate["scientific_config"]["motion"]["recipe"] == (
            MOTION_RECIPE
        )
        assert "motion_corrected_recording" not in estimate["stage_seconds"]

        applied = run_v2_pipeline(**inputs, motion_mode="apply", **motion)
        sorting_ids.append(applied["sorting_id"])
        assert applied["sorting_id"] != off["sorting_id"]
        assert applied["motion_estimate_id"] == expected_estimate
        assert applied["motion_estimate_status"] == "reused"
        assert applied["motion_corrected_recording_status"] == "computed"
        corrected_key = {
            "motion_corrected_recording_id": applied[
                "motion_corrected_recording_id"
            ]
        }
        corrected_row = (MotionCorrectedRecording & corrected_key).fetch1()
        assert (
            SortingSelection.resolve_motion_correction(
                {"sorting_id": applied["sorting_id"]}
            )
            == applied["motion_corrected_recording_id"]
        )
        assert Sorting & {"sorting_id": applied["sorting_id"]}
        # The shipped recipe extrapolates at the borders: nothing removed.
        assert applied["motion_removed_channel_ids"] == []
        assert list(corrected_row["removed_channel_ids"]) == []
        assert applied["motion_estimation_preset"] == "dredge_fast"
        setup = applied["scientific_config"]["motion"]
        assert (setup["mode"], setup["estimation_preset"]) == (
            "apply",
            "dredge_fast",
        )
        assert setup["border_mode"] == "force_extrapolate"

        rerun = run_v2_pipeline(**inputs, motion_mode="apply", **motion)
        for field in (
            "sorting_id",
            "motion_estimate_id",
            "motion_corrected_recording_id",
        ):
            assert rerun[field] == applied[field]
        assert {
            rerun[f"{stage}_status"]
            for stage in (
                "motion_estimate",
                "motion_corrected_recording",
                "sorting",
            )
        } == {"reused"}
    finally:
        drop_pipeline_sorts(sorting_ids)


def test_pipeline_motion_apply_on_a_concatenation(discontinuous_sources):
    """A concat run in ``apply`` mode estimates the concatenation it built,
    corrects it and sorts the corrected recording, with the same receipt."""
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecordingSelection,
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.sorting import SortingSelection

    summary = None
    try:
        summary = run_v2_pipeline(
            concat_session_group_owner=MOTION_TEAM,
            concat_session_group_name=CONCAT_GROUP,
            pipeline_preset=PIPELINE_PRESET,
            motion_mode="apply",
            motion_correction_params_name=MOTION_RECIPE,
        )
        assert summary["source_mode"] == "concat"
        concat_key = {"concat_recording_id": summary["concat_recording_id"]}
        lineage = MotionEstimateSelection.resolve_source(
            {"motion_estimate_id": summary["motion_estimate_id"]}
        )
        assert (lineage.kind, lineage.key) == (
            "concatenated_recording",
            concat_key,
        )
        assert (
            MotionCorrectedRecordingSelection
            & {
                "motion_corrected_recording_id": summary[
                    "motion_corrected_recording_id"
                ]
            }
        ).fetch1("motion_estimate_id") == summary["motion_estimate_id"]
        sort_key = {"sorting_id": summary["sorting_id"]}
        assert SortingSelection.resolve_motion_correction(sort_key) == (
            summary["motion_corrected_recording_id"]
        )
        effective = SortingSelection.resolve_effective_source(sort_key)
        assert effective.lineage.key == concat_key
        assert effective.traces.kind == "motion_corrected_recording"
        assert summary["motion_estimation_preset"] == "dredge_fast"
        assert summary["motion_removed_channel_ids"] == []
        assert summary["motion_estimate_status"] == "computed"
        assert set(summary["member_merge_ids"]) == {0, 1}
        setup = summary["scientific_config"]["motion"]
        assert (setup["mode"], setup["recipe"]) == ("apply", MOTION_RECIPE)
        assert "concatenation" in setup["description"]
    finally:
        if summary is not None:
            drop_pipeline_sorts([summary["sorting_id"]])
            drop_motion_selections(
                {"concat_recording_id": summary["concat_recording_id"]}
            )


def test_pipeline_surfaces_a_span_without_evidence(
    drift_recording, discontinuous_sources
):
    """An artifact exclusion covering the whole short second acquisition span
    of a gapped recording leaves that span without a kept peak. The run is
    not refused: the receipt lists the span (frames and source times, as
    ``MotionEstimate.get_spans_without_evidence`` reports it) and the run
    warns. An estimate whose spans all kept peaks reports none."""
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.motion import (
        MotionEstimate,
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from tests.spikesorting.v2._ingest_helpers import configure_v2_run_inputs

    nwb_file_name = drift_recording["nwb_file_name"]
    t0 = discontinuous_sources["t0"]
    with_evidence = populated_estimate(
        recording_id=discontinuous_sources["member_b"]["recording_id"]
    )
    assert MotionEstimate().get_spans_without_evidence(with_evidence) == []

    # The second span (1.5 s, above the recipe's 1 s minimum segment) is
    # excluded whole; the exclusion masks ~11% of the recording, well under
    # the artifact guard's 50%.
    interval_list_name = "motion short second span"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
            "valid_times": np.array(
                [[t0 + 2.0, t0 + 14.0], [t0 + 15.0, t0 + 16.5]]
            ),
            "pipeline": "motion_estimate_test",
        },
        skip_duplicates=True,
    )
    summary = None
    try:
        summary = run_v2_pipeline(
            **configure_v2_run_inputs(
                nwb_file_name,
                MOTION_TEAM,
                interval_list_name=interval_list_name,
            ),
            pipeline_preset=PIPELINE_PRESET,
            motion_mode="estimate",
            motion_correction_params_name=MOTION_RECIPE,
            manual_excluded_times=[[t0 + 14.5, t0 + 16.9]],
        )
        estimate_key = {"motion_estimate_id": summary["motion_estimate_id"]}
        row = (MotionEstimate & estimate_key).fetch1()
        assert row["peaks_per_continuity_span"][0] > 0
        assert row["peaks_per_continuity_span"][1] == 0
        spans = MotionEstimate().get_spans_without_evidence(estimate_key)
        assert summary["motion_spans_without_evidence"] == spans
        assert len(spans) == 1
        assert spans[0]["span_index"] == 1
        assert (spans[0]["start_frame"], spans[0]["end_frame"]) == tuple(
            row["continuity_spans"][1]
        )
        assert spans[0]["source_start_s"] == pytest.approx(t0 + 15.0, abs=1e-3)
        assert spans[0]["source_end_s"] == pytest.approx(t0 + 16.5, abs=1e-3)
        warnings = [w for w in summary["warnings"] if "kept no peaks" in str(w)]
        assert len(warnings) == 1
        assert str(summary["motion_estimate_id"]) in warnings[0]
    finally:
        if summary is not None:
            drop_pipeline_sorts([summary["sorting_id"]])
            (
                MotionEstimateSelection
                & {"motion_estimate_id": summary["motion_estimate_id"]}
            ).super_delete(warn=False, safemode=False)


@pytest.mark.parametrize(
    "failing_stage, target, offset_s",
    [
        ("motion_estimate", "estimate_motion_in_spans", 2.0),
        ("motion_corrected_recording", "apply_motion_on_estimation_clock", 4.0),
    ],
)
def test_motion_stage_failure_stops_before_sorting(
    drift_recording, monkeypatch, failing_stage, target, offset_s
):
    """An estimation or application failure fails its own stage: no estimate
    or corrected row is written for it, and no sort is selected or run --
    the run never falls back to the uncorrected source."""
    from spyglass.spikesorting.v2 import _motion
    from spyglass.spikesorting.v2.exceptions import PipelineStageError
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline

    inputs = _pipeline_inputs(drift_recording)
    t0 = session_start_s(drift_recording["nwb_file_name"])
    # A mask of its own gives this case a fresh estimate (no cached reuse).
    exclusion = [[t0 + offset_s, t0 + offset_s + 0.5]]

    def _fail(*_args, **_kwargs):
        raise ValueError(f"planted {failing_stage} failure")

    monkeypatch.setattr(_motion, target, _fail)
    before = _row_counts()
    with pytest.raises(PipelineStageError) as raised:
        run_v2_pipeline(
            **inputs,
            manual_excluded_times=exclusion,
            motion_mode="apply",
            motion_correction_params_name=MOTION_RECIPE,
        )
    assert raised.value.stage == failing_stage
    assert f"planted {failing_stage} failure" in str(raised.value)
    after = _row_counts()
    expected = dict(before)
    if failing_stage == "motion_corrected_recording":
        # The estimate succeeded; its corrected recording was only selected.
        expected["MotionEstimate"] += 1
        expected["MotionCorrectedRecordingSelection"] += 1
    assert after == expected
    partial = raised.value.partial_run_summary
    assert partial["motion_corrected_recording_id"] is None
    assert "sorting_id" not in partial


def test_motion_selection_failure_is_a_stage_error(
    drift_recording, monkeypatch
):
    """A refused motion selection insert fails its stage like a failed
    populate -- a ``PipelineStageError`` carrying the partial run summary,
    not a bare error -- and nothing is estimated or sorted."""
    from spyglass.spikesorting.v2.exceptions import PipelineStageError
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline

    def _refuse(cls, key):
        raise ValueError("planted selection refusal")

    monkeypatch.setattr(
        MotionEstimateSelection, "insert_selection", classmethod(_refuse)
    )
    before = _row_counts()
    with pytest.raises(PipelineStageError) as raised:
        run_v2_pipeline(
            **_pipeline_inputs(drift_recording),
            motion_mode="apply",
            motion_correction_params_name=MOTION_RECIPE,
        )
    assert raised.value.stage == "motion_estimate"
    assert raised.value.original_type == "ValueError"
    assert "planted selection refusal" in str(raised.value)
    partial = raised.value.partial_run_summary
    assert partial["recording_id"] is not None
    assert partial["motion_estimate_id"] is None
    assert "sorting_id" not in partial
    assert _row_counts() == before


def test_preflight_previews_the_motion_ids_the_run_mints(drift_recording):
    """For ``apply``, preflight's expected estimate, corrected-recording and
    sort ids are the ones the run then produces, and every motion check
    passes on the eligible polymer shank with a sorter that does not correct
    motion itself."""
    from spyglass.spikesorting.v2.pipeline import (
        preflight_v2_pipeline,
        run_v2_pipeline,
    )

    inputs = _pipeline_inputs(drift_recording)
    motion = {
        "motion_mode": "apply",
        "motion_correction_params_name": MOTION_RECIPE,
    }
    report = preflight_v2_pipeline(**inputs, **motion)
    assert report.ok, report.errors
    checks = {c.name: c.ok for c in report.checks}
    for name in (
        "motion_request_valid",
        "motion_recipe_exists",
        "motion_geometry_supported",
        "sorter_motion_correction_off",
    ):
        assert checks[name] is True
    ids = report.expected_ids
    assert list(ids) == [
        "recording_id",
        "artifact_detection_id",
        "motion_estimate_id",
        "motion_corrected_recording_id",
        "sorting_id",
    ]
    assert "motion_corrected_recording:" in report.summary()
    summary = None
    try:
        summary = run_v2_pipeline(**inputs, **motion)
        for name in (
            "motion_estimate_id",
            "motion_corrected_recording_id",
            "sorting_id",
        ):
            assert ids[name]["id"] == summary[name]
    finally:
        if summary is not None:
            drop_pipeline_sorts([summary["sorting_id"]])


def _saved_estimate_of_the_run(drift_recording) -> dict:
    """The ``dredge_fast_v1`` estimate of the pipeline run's recording under
    the run's own artifact detection, saved through the tables."""
    from spyglass.spikesorting.v2._pipeline_presets import _PIPELINE_PRESETS
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )

    recording_id = drift_recording["recording_key"]["recording_id"]
    artifact_key = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": recording_id,
            "artifact_detection_params_name": _PIPELINE_PRESETS[
                PIPELINE_PRESET
            ].artifact_detection_params_name,
            "manual_excluded_times": [],
        }
    )
    RecordingArtifactDetection.populate(artifact_key, reserve_jobs=False)
    return {
        **populated_estimate(recording_id=recording_id, **artifact_key),
        **artifact_key,
    }


def test_apply_reuses_a_supplied_estimate(drift_recording, monkeypatch):
    """``apply`` with ``motion_estimate_id`` corrects and sorts with exactly
    that saved estimate: no estimate is computed (the estimator is patched to
    fail), the corrected recording is made from it, and the receipt,
    ``describe_run`` and preflight's preview all name it."""
    from spyglass.spikesorting.v2 import _motion
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecordingSelection,
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.pipeline import (
        describe_run,
        preflight_v2_pipeline,
        run_v2_pipeline,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    inputs = _pipeline_inputs(drift_recording)
    saved = _saved_estimate_of_the_run(drift_recording)
    estimate_id = saved["motion_estimate_id"]
    motion = {
        "motion_mode": "apply",
        "motion_correction_params_name": MOTION_RECIPE,
        # A string id is accepted like the UUID.
        "motion_estimate_id": str(estimate_id),
    }
    report = preflight_v2_pipeline(**inputs, **motion)
    assert report.ok, report.errors
    assert {c.name: c.ok for c in report.checks}["motion_estimate_applicable"]
    ids = report.expected_ids
    assert ids["artifact_detection_id"]["id"] == saved["artifact_detection_id"]
    assert ids["motion_estimate_id"]["id"] == estimate_id

    def _no_estimation(*_args, **_kwargs):
        raise AssertionError("a supplied estimate was estimated again")

    def _no_selection(cls, key):
        raise AssertionError("a supplied estimate was selected again")

    monkeypatch.setattr(_motion, "estimate_motion_in_spans", _no_estimation)
    monkeypatch.setattr(
        MotionEstimateSelection, "insert_selection", classmethod(_no_selection)
    )
    estimates_before = _row_counts()["MotionEstimate"]
    summary = None
    try:
        summary = run_v2_pipeline(**inputs, **motion)
        assert summary["motion_estimate_id"] == estimate_id
        assert summary["motion_estimate_supplied"] is True
        assert summary["motion_estimate_status"] == "reused"
        assert summary["motion_estimation_preset"] == "dredge_fast"
        assert _row_counts()["MotionEstimate"] == estimates_before
        corrected_id = summary["motion_corrected_recording_id"]
        assert (
            MotionCorrectedRecordingSelection
            & {"motion_corrected_recording_id": corrected_id}
        ).fetch1("motion_estimate_id") == estimate_id
        assert (
            SortingSelection.resolve_motion_correction(
                {"sorting_id": summary["sorting_id"]}
            )
            == corrected_id
        )
        for name in ("motion_corrected_recording_id", "sorting_id"):
            assert ids[name]["id"] == summary[name]
        config = describe_run(summary).set_index("setting")["value"]
        assert config["motion_estimate_supplied"] == "True"
        assert config["motion_estimate_id"] == str(estimate_id)
    finally:
        if summary is not None:
            drop_pipeline_sorts([summary["sorting_id"]])


@pytest.mark.parametrize(
    "case, preflight",
    [
        ("other_source", True),
        ("other_mask", True),
        ("other_estimation_row", True),
        ("concat_run", True),
        ("other_mask", False),
        ("other_estimation_row", False),
    ],
)
def test_a_supplied_estimate_that_does_not_match_fails_before_sorting(
    drift_recording, discontinuous_sources, case, preflight
):
    """A supplied estimate of another recording, of the run's recording under
    another mask, made with an estimation row other than the recipe's, or of
    a recording for a concat run, is refused -- by preflight, or with
    ``preflight=False`` by the ``motion_estimate`` stage -- naming both
    values, before anything is corrected or sorted."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        PipelineStageError,
        PreflightError,
    )
    from spyglass.spikesorting.v2.motion import MotionCorrectionParameters
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline

    MotionCorrectionParameters.insert_default()
    run_recording_id = drift_recording["recording_key"]["recording_id"]
    saved = _saved_estimate_of_the_run(drift_recording)
    inputs = _pipeline_inputs(drift_recording)
    recipe = MOTION_RECIPE
    if case == "other_source":
        other_id = discontinuous_sources["member_b"]["recording_id"]
        estimate = populated_estimate(recording_id=other_id)
        expected = [str(other_id), str(run_recording_id)]
    elif case == "other_mask":
        estimate = populated_estimate(recording_id=run_recording_id)
        expected = [
            f"artifact_detection_id None != the run's "
            f"{saved['artifact_detection_id']}"
        ]
    elif case == "other_estimation_row":
        estimate = saved
        recipe = "dredge_v1"
        expected = ["'dredge_fast_v1'", "'dredge_v1'"]
    else:
        estimate = saved
        inputs = {
            "concat_session_group_owner": MOTION_TEAM,
            "concat_session_group_name": CONCAT_GROUP,
            "pipeline_preset": PIPELINE_PRESET,
        }
        expected = [str(run_recording_id), "sorts a concatenation"]

    before = _row_counts()
    curations_before = len(CurationV2())
    raised_type = PreflightError if preflight else PipelineStageError
    with pytest.raises(raised_type) as raised:
        run_v2_pipeline(
            **inputs,
            preflight=preflight,
            motion_mode="apply",
            motion_correction_params_name=recipe,
            motion_estimate_id=estimate["motion_estimate_id"],
        )
    message = str(raised.value)
    assert f"motion estimate {estimate['motion_estimate_id']}" in message
    for fragment in expected:
        assert fragment in message
    if not preflight:
        assert raised.value.stage == "motion_estimate"
        partial = raised.value.partial_run_summary
        assert partial["motion_estimate_supplied"] is True
        assert "sorting_id" not in partial
    assert _row_counts() == before
    assert len(CurationV2()) == curations_before


def _drop_estimate(motion_estimate_id) -> None:
    """Delete one motion-estimate selection (and its estimate) by id, after
    the sorts that read its corrected recordings."""
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection
    from tests.spikesorting.v2._motion_db_helpers import (
        drop_sorts_of_estimates,
    )

    key = [{"motion_estimate_id": motion_estimate_id}]
    drop_sorts_of_estimates(key)
    (MotionEstimateSelection & key).super_delete(warn=False, safemode=False)


def test_estimate_motion_then_apply_that_estimate(drift_recording, monkeypatch):
    """``estimate_motion`` builds the source ``run_v2_pipeline`` would sort,
    saves its estimate (the id preflight previews for that run) and selects
    no sort or curation; the receipt's diagnostics are the saved row's, and
    a second call reuses the estimate. ``apply`` with that id then corrects
    and sorts with it without estimating or selecting an estimate again."""
    from spyglass.spikesorting.v2 import _motion
    from spyglass.spikesorting.v2.artifact import RecordingArtifactSelection
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecordingSelection,
        MotionEstimate,
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.pipeline import (
        estimate_motion,
        preflight_v2_pipeline,
        run_v2_pipeline,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    inputs = _pipeline_inputs(drift_recording)
    t0 = session_start_s(drift_recording["nwb_file_name"])
    # A mask of its own gives this test a fresh estimate (no cached reuse).
    exclusion = [[t0 + 6.0, t0 + 6.5]]
    before = _row_counts()
    curations_before = len(CurationV2())
    receipt = None
    try:
        receipt = estimate_motion(
            **inputs,
            manual_excluded_times=exclusion,
            motion_correction_params_name=MOTION_RECIPE,
        )
        expected = dict(before)
        expected["MotionEstimate"] += 1
        assert _row_counts() == expected
        assert len(CurationV2()) == curations_before
        assert "sorting_id" not in receipt

        estimate_key = {"motion_estimate_id": receipt["motion_estimate_id"]}
        lineage = MotionEstimateSelection.resolve_source(estimate_key)
        assert lineage.key == {"recording_id": receipt["recording_id"]}
        assert lineage.artifact_detection_id == receipt["artifact_detection_id"]
        assert receipt["artifact_detection_id"] is not None
        report = preflight_v2_pipeline(
            **inputs,
            manual_excluded_times=exclusion,
            motion_mode="estimate",
            motion_correction_params_name=MOTION_RECIPE,
        )
        assert report.expected_ids["motion_estimate_id"]["id"] == (
            receipt["motion_estimate_id"]
        )
        assert receipt["source_mode"] == "single_session"
        assert receipt["motion_correction_params_name"] == MOTION_RECIPE
        assert receipt["motion_estimation_preset"] == "dredge_fast"
        assert receipt["motion_estimate_status"] == "computed"
        assert receipt["motion_spans_without_evidence"] == []
        row = (MotionEstimate & estimate_key).fetch1()
        assert receipt["motion_diagnostics"] == {
            name: row[name]
            for name in (
                "n_peaks_detected",
                "n_peaks_kept",
                "max_abs_displacement_um",
                "n_temporal_bins",
            )
        }
        # A +/-25 um drift was planted; the estimate must see it.
        assert receipt["motion_diagnostics"]["n_peaks_kept"] > 0
        assert receipt["motion_diagnostics"]["max_abs_displacement_um"] > 10.0
        # The receipt itself selects the estimate to report; the report
        # shows the exclusion as masked.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, summary = MotionEstimate().report(receipt)
        plt.close("all")
        assert summary["motion_estimate_id"] == receipt["motion_estimate_id"]
        assert summary["spans_without_evidence"] == (
            receipt["motion_spans_without_evidence"]
        )
        assert summary["max_abs_displacement_um"] == (
            receipt["motion_diagnostics"]["max_abs_displacement_um"]
        )
        assert any(
            interval["source_start_s"] <= t0 + 6.0 + 1e-3
            and interval["source_end_s"] >= t0 + 6.5 - 1e-3
            for interval in summary["masked_intervals"]
        )
        assert set(receipt["stage_seconds"]) == {
            "recording",
            "artifact_detection",
            "motion_estimate",
        }
        assert "scientific_config" not in receipt
        assert "sorter_config" not in receipt

        again = estimate_motion(
            **inputs,
            manual_excluded_times=exclusion,
            motion_correction_params_name=MOTION_RECIPE,
        )
        assert again["motion_estimate_id"] == receipt["motion_estimate_id"]
        assert again["motion_estimate_status"] == "reused"
        assert _row_counts() == expected

        def _no_estimation(*_args, **_kwargs):
            raise AssertionError("the inspected estimate was estimated again")

        def _no_selection(cls, key):
            raise AssertionError("the inspected estimate was selected again")

        monkeypatch.setattr(_motion, "estimate_motion_in_spans", _no_estimation)
        monkeypatch.setattr(
            MotionEstimateSelection,
            "insert_selection",
            classmethod(_no_selection),
        )
        applied = run_v2_pipeline(
            **inputs,
            manual_excluded_times=exclusion,
            motion_mode="apply",
            motion_correction_params_name=MOTION_RECIPE,
            motion_estimate_id=receipt["motion_estimate_id"],
        )
        assert applied["motion_estimate_id"] == receipt["motion_estimate_id"]
        assert applied["motion_estimate_supplied"] is True
        assert applied["motion_estimate_status"] == "reused"
        assert applied["recording_id"] == receipt["recording_id"]
        assert applied["artifact_detection_id"] == (
            receipt["artifact_detection_id"]
        )
        assert _row_counts()["MotionEstimate"] == expected["MotionEstimate"]
        corrected_key = {
            "motion_corrected_recording_id": applied[
                "motion_corrected_recording_id"
            ]
        }
        assert (MotionCorrectedRecordingSelection & corrected_key).fetch1(
            "motion_estimate_id"
        ) == receipt["motion_estimate_id"]
        assert (
            SortingSelection.resolve_motion_correction(
                {"sorting_id": applied["sorting_id"]}
            )
            == corrected_key["motion_corrected_recording_id"]
        )
    finally:
        if receipt is not None:
            _drop_estimate(receipt["motion_estimate_id"])
            # The exclusion's own artifact detection (and its merge entry).
            _drop_artifact_detections([receipt["artifact_detection_id"]])
    assert not (
        RecordingArtifactSelection
        & {"artifact_detection_id": receipt["artifact_detection_id"]}
    )


def test_estimate_motion_then_apply_on_a_concatenation(discontinuous_sources):
    """On a concat source, ``estimate_motion`` estimates the concatenation a
    concat run builds (and sorts nothing), and ``apply`` with that id sorts
    the concatenation corrected with it."""
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecordingSelection,
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.pipeline import (
        estimate_motion,
        run_v2_pipeline,
    )

    concat = {
        "concat_session_group_owner": MOTION_TEAM,
        "concat_session_group_name": CONCAT_GROUP,
        "pipeline_preset": PIPELINE_PRESET,
    }
    before = _row_counts()
    receipt = applied = None
    try:
        receipt = estimate_motion(
            **concat, motion_correction_params_name=MOTION_RECIPE
        )
        assert receipt["source_mode"] == "concat"
        assert "sorting_id" not in receipt
        for table in ("SortingSelection", "Sorting"):
            assert _row_counts()[table] == before[table]
        lineage = MotionEstimateSelection.resolve_source(
            {"motion_estimate_id": receipt["motion_estimate_id"]}
        )
        assert (lineage.kind, lineage.key) == (
            "concatenated_recording",
            {"concat_recording_id": receipt["concat_recording_id"]},
        )
        assert receipt["motion_diagnostics"]["n_peaks_kept"] > 0

        applied = run_v2_pipeline(
            **concat,
            motion_mode="apply",
            motion_correction_params_name=MOTION_RECIPE,
            motion_estimate_id=receipt["motion_estimate_id"],
        )
        assert applied["concat_recording_id"] == receipt["concat_recording_id"]
        assert applied["motion_estimate_id"] == receipt["motion_estimate_id"]
        assert applied["motion_estimate_status"] == "reused"
        assert (
            MotionCorrectedRecordingSelection
            & {
                "motion_corrected_recording_id": applied[
                    "motion_corrected_recording_id"
                ]
            }
        ).fetch1("motion_estimate_id") == receipt["motion_estimate_id"]
    finally:
        if applied is not None:
            drop_pipeline_sorts([applied["sorting_id"]])
        if receipt is not None:
            drop_motion_selections(
                {"concat_recording_id": receipt["concat_recording_id"]}
            )


def test_concat_preflight_names_the_calling_entry_point(dj_conn):
    """A concat preflight failure names the entry point that ran it."""
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.pipeline import estimate_motion

    with pytest.raises(PreflightError, match="^estimate_motion: SessionGroup"):
        estimate_motion(
            concat_session_group_owner="no_such_owner",
            concat_session_group_name="no_such_group",
            pipeline_preset=PIPELINE_PRESET,
            motion_correction_params_name=MOTION_RECIPE,
        )


def test_estimate_motion_does_not_need_the_sorter(
    drift_recording, discontinuous_sources, monkeypatch
):
    """With the preset's sorter reported unavailable, ``estimate_motion``
    and both preflights without the sorter-only checks pass, while a
    ``run_v2_pipeline`` call (which sorts) is still refused."""
    from spyglass.spikesorting.v2 import _pipeline_preflight
    from spyglass.spikesorting.v2._pipeline_presets import _PIPELINE_PRESETS
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.pipeline import (
        estimate_motion,
        preflight_v2_pipeline,
        run_v2_pipeline,
    )

    def _unavailable(bundle, sis, non_si_sorters, check):
        check("sorter_installed", False, "planted: the sorter is unavailable")

    monkeypatch.setattr(
        _pipeline_preflight, "_check_local_sorter_runtime", _unavailable
    )
    inputs = _pipeline_inputs(drift_recording)
    motion = {
        "motion_mode": "estimate",
        "motion_correction_params_name": MOTION_RECIPE,
    }
    report = preflight_v2_pipeline(**inputs, **motion, sort_checks=False)
    assert report.ok, report.errors
    assert not {c.name for c in report.checks} & {
        "sorter_params_exist",
        "sorter_params_valid",
        "analyzer_waveform_params_exist",
        "sampling_rate_matches",
        "sorter_installed",
        "container_runtime_available",
    }
    with pytest.raises(PreflightError, match="planted"):
        run_v2_pipeline(**inputs, **motion)
    receipt = estimate_motion(
        **inputs, motion_correction_params_name=MOTION_RECIPE
    )
    assert receipt["motion_estimate_id"] is not None

    bundle = _PIPELINE_PRESETS[PIPELINE_PRESET]
    concat = (MOTION_TEAM, CONCAT_GROUP, bundle)
    # Raises nothing (it returns its advisories).
    _pipeline_preflight.assert_concat_preflight(
        *concat, **motion, sort_checks=False
    )
    with pytest.raises(PreflightError, match="planted"):
        _pipeline_preflight.assert_concat_preflight(*concat, **motion)


def test_concat_preflight_refuses_an_estimate_of_other_member_masks(
    discontinuous_sources,
):
    """For a concat run, preflight compares a supplied estimate's frozen
    members with the member recordings and masks the run would select: the
    estimate of the concatenation it builds passes, and the same
    estimate for a run that adds a manual exclusion to member 0 is refused,
    naming member 0's two detections, before anything is built or sorted."""
    from spyglass.spikesorting.v2 import _pipeline_preflight
    from spyglass.spikesorting.v2._pipeline_presets import _PIPELINE_PRESETS
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.pipeline import (
        estimate_motion,
        run_v2_pipeline,
    )

    concat = {
        "concat_session_group_owner": MOTION_TEAM,
        "concat_session_group_name": CONCAT_GROUP,
        "pipeline_preset": PIPELINE_PRESET,
    }
    motion = {
        "motion_mode": "apply",
        "motion_correction_params_name": MOTION_RECIPE,
    }
    t0 = discontinuous_sources["t0"]
    receipt = None
    try:
        receipt = estimate_motion(
            **concat, motion_correction_params_name=MOTION_RECIPE
        )
        estimate_id = receipt["motion_estimate_id"]
        _pipeline_preflight.assert_concat_preflight(
            MOTION_TEAM,
            CONCAT_GROUP,
            _PIPELINE_PRESETS[PIPELINE_PRESET],
            **motion,
            motion_estimate_id=estimate_id,
        )
        frozen = {
            row["member_index"]: row["artifact_detection_id"]
            for row in receipt["member_artifacts"]
        }
        before = _row_counts()
        with pytest.raises(PreflightError) as raised:
            run_v2_pipeline(
                **concat,
                **motion,
                motion_estimate_id=estimate_id,
                # Inside member 0's interval (session seconds 16-20).
                manual_excluded_times={0: [[t0 + 17.0, t0 + 17.5]]},
            )
        message = str(raised.value)
        assert f"motion estimate {estimate_id}" in message
        assert f"member 0: its artifact_detection_id {frozen[0]} != " in message
        assert "member 1:" not in message
        assert _row_counts() == before
    finally:
        if receipt is not None:
            drop_motion_selections(
                {"concat_recording_id": receipt["concat_recording_id"]}
            )


def test_preflight_refuses_a_supplied_estimate_of_changed_traces(
    drift_recording,
):
    """A supplied estimate whose source recording's content hash changed
    since it was selected describes other traces: preflight refuses it,
    naming both hashes."""
    import datajoint as dj

    from spyglass.spikesorting.v2.pipeline import preflight_v2_pipeline
    from spyglass.spikesorting.v2.recording import Recording

    saved = _saved_estimate_of_the_run(drift_recording)
    recording_key = dict(drift_recording["recording_key"])
    recordings = dj.FreeTable(dj.conn(), Recording.full_table_name)
    original = (recordings & recording_key).fetch1("content_hash")
    changed = "0" * 64
    recordings.update1({**recording_key, "content_hash": changed})
    try:
        report = preflight_v2_pipeline(
            **_pipeline_inputs(drift_recording),
            motion_mode="apply",
            motion_correction_params_name=MOTION_RECIPE,
            motion_estimate_id=saved["motion_estimate_id"],
        )
    finally:
        recordings.update1({**recording_key, "content_hash": original})
    assert not report.ok
    (fix,) = [
        check.fix
        for check in report.checks
        if check.name == "motion_estimate_applicable"
    ]
    assert "changed since motion estimate" in fix
    assert f"content_hash {changed} != {original}" in fix


@pytest.fixture
def self_correcting_preset(dj_conn):
    """A registered preset whose sorter row (spykingcircus2 ``default``)
    runs the sorter's own motion correction."""
    from spyglass.spikesorting.v2._pipeline_presets import (
        _PIPELINE_PRESETS,
        register_pipeline_preset,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters

    name = "motion_test_spykingcircus2_2026_09"
    SorterParameters.insert_default()
    register_pipeline_preset(
        name,
        _PIPELINE_PRESETS[PIPELINE_PRESET].model_copy(
            update={
                "sorter": "spykingcircus2",
                "sorter_params_name": "default",
                "sorter_family": "spykingcircus2",
            }
        ),
    )
    yield name
    _PIPELINE_PRESETS.pop(name, None)


@pytest.fixture
def short_sort_group(drift_recording):
    """A four-contact sort group (78 um, under the shipped recipes' 80 um
    detection radius) on the drift session, removed afterwards."""
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = drift_recording["nwb_file_name"]
    inputs = _pipeline_inputs(drift_recording)
    key = {"nwb_file_name": nwb_file_name, "sort_group_id": 99}
    electrodes = (
        Electrode
        & {"nwb_file_name": nwb_file_name}
        & (
            SortGroupV2.SortGroupElectrode
            & {**key, "sort_group_id": inputs["sort_group_id"]}
        ).proj()
    ).fetch("KEY", as_dict=True, order_by="electrode_id", limit=4)
    SortGroupV2.insert1({**key, "reference_mode": "none"})
    SortGroupV2.SortGroupElectrode.insert(
        [{**key, **electrode} for electrode in electrodes]
    )
    yield key
    (SortGroupV2 & key).super_delete(warn=False, safemode=False)


def _drop_artifact_detections(artifact_ids) -> None:
    """Delete artifact detections with the motion selections on their masks.

    Motion-estimate selections (and, by cascade, their estimates and
    corrected-recording selections) go first, then the ``ArtifactDetectionOutput``
    merge masters, then the split detection rows. Neither writes an analysis
    NWB file, so nothing is orphaned on disk.
    """
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import (
        ArtifactDetectionOutput,
    )
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    keys = [{"artifact_detection_id": a} for a in set(artifact_ids) if a]
    if not keys:
        return
    merge_ids = (ArtifactDetectionOutput.RecordingSource & keys).fetch(
        "merge_id"
    )
    merges = [{"merge_id": m} for m in merge_ids]
    if merges:
        estimates = (
            MotionEstimateSelection.ArtifactDetectionSource
            & [{"artifact_detection_merge_id": m} for m in merge_ids]
        ).fetch("KEY", as_dict=True)
        if estimates:
            (MotionEstimateSelection & estimates).super_delete(
                warn=False, safemode=False
            )
        (ArtifactDetectionOutput & merges).super_delete(
            warn=False, safemode=False, force_masters=True
        )
    (RecordingArtifactDetection & keys).super_delete(warn=False, safemode=False)
    (RecordingArtifactSelection & keys).super_delete(warn=False, safemode=False)


def test_invalid_support_and_geometry_fail_before_sorting(
    drift_recording, monkeypatch, self_correcting_preset, short_sort_group
):
    """Each unsupported case fails clearly before any sort, and no corrected
    recording or sort row is written: an estimate with no valid evidence, a
    non-finite estimate, a correction that removes every channel, a sort
    group too short for the recipe, a sorter row that corrects motion itself,
    and a recipe name with no row (with preflight on or off)."""
    import spikeinterface.sortingcomponents.motion as si_motion

    from spyglass.spikesorting.v2._params.motion_estimation import (
        MotionEstimationParamsSchema,
    )
    from spyglass.spikesorting.v2._recipe_catalog import (
        KRIGING_FORCE_EXTRAPOLATE,
        KRIGING_REMOVE_CHANNELS,
    )
    from spyglass.spikesorting.v2.artifact import RecordingArtifactSelection
    from spyglass.spikesorting.v2.exceptions import (
        PipelineStageError,
        PreflightError,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectionParameters,
        MotionEstimationParameters,
    )
    from spyglass.spikesorting.v2.pipeline import (
        preflight_v2_pipeline,
        run_v2_pipeline,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection

    inputs = _pipeline_inputs(drift_recording)
    t0 = session_start_s(drift_recording["nwb_file_name"])
    test_recipes = ["no_evidence_test", "remove_channels_test"]
    artifact_ids = []
    MotionEstimationParameters.insert1(
        {
            "motion_estimation_params_name": "no_evidence_test",
            "params": MotionEstimationParamsSchema(
                preset="dredge_fast",
                max_gap_s=30.0,
                detect_kwargs={"detect_threshold": 1.0e6},
            ).model_dump(),
        },
        skip_duplicates=True,
    )
    MotionCorrectionParameters.insert(
        [
            {
                "motion_correction_params_name": "no_evidence_test",
                "motion_estimation_params_name": "no_evidence_test",
                "motion_interpolation_params_name": KRIGING_FORCE_EXTRAPOLATE,
            },
            {
                "motion_correction_params_name": "remove_channels_test",
                "motion_estimation_params_name": MOTION_RECIPE,
                "motion_interpolation_params_name": KRIGING_REMOVE_CHANNELS,
            },
        ],
        skip_duplicates=True,
    )
    real_estimate_motion = si_motion.estimate_motion

    def _shifted_estimate(shift_um):
        def _estimate(*args, **kwargs):
            motion = real_estimate_motion(*args, **kwargs)
            motion.displacement[0] = motion.displacement[0] + shift_um
            return motion

        return _estimate

    runtime_cases = [
        # (recipe, planted estimator, failing stage, message, offset s)
        ("no_evidence_test", None, "motion_estimate", "no valid evidence", 8),
        (
            MOTION_RECIPE,
            _shifted_estimate(np.nan),
            "motion_estimate",
            "non-finite",
            10,
        ),
        (
            "remove_channels_test",
            _shifted_estimate(5000.0),
            "motion_corrected_recording",
            "removed every channel",
            12,
        ),
    ]
    try:
        for recipe, estimator, stage, message, offset_s in runtime_cases:
            if estimator is not None:
                monkeypatch.setattr(si_motion, "estimate_motion", estimator)
            before = _row_counts()
            with pytest.raises(PipelineStageError) as raised:
                run_v2_pipeline(
                    **inputs,
                    # A mask of its own gives each case a fresh estimate.
                    manual_excluded_times=[
                        [t0 + offset_s, t0 + offset_s + 0.5]
                    ],
                    motion_mode="apply",
                    motion_correction_params_name=recipe,
                )
            monkeypatch.undo()
            artifact_ids.append(
                raised.value.partial_run_summary["artifact_detection_id"]
            )
            assert raised.value.stage == stage, recipe
            assert message in str(raised.value), recipe
            after = _row_counts()
            for table in (
                "MotionCorrectedRecording",
                "SortingSelection",
                "Sorting",
            ):
                assert after[table] == before[table], (recipe, table)

        short_inputs = {
            **inputs,
            "sort_group_id": short_sort_group["sort_group_id"],
        }
        preflight_cases = [
            # (request, recipe, failing check, message)
            (
                short_inputs,
                MOTION_RECIPE,
                "motion_geometry_supported",
                "detection radius_um",
            ),
            (
                {**inputs, "pipeline_preset": self_correcting_preset},
                MOTION_RECIPE,
                "sorter_motion_correction_off",
                "apply_motion_correction=False",
            ),
            (inputs, "no_such_recipe", "motion_recipe_exists", "is missing"),
        ]
        for request, recipe, check, message in preflight_cases:
            report = preflight_v2_pipeline(
                **request,
                motion_mode="apply",
                motion_correction_params_name=recipe,
            )
            failed = {c.name: c.fix for c in report.checks if not c.ok}
            assert message in failed[check], failed
            before = _row_counts()
            with pytest.raises(PreflightError, match=message):
                run_v2_pipeline(
                    **request,
                    motion_mode="apply",
                    motion_correction_params_name=recipe,
                )
            assert _row_counts() == before

        # With preflight off, a missing recipe still fails before any stage:
        # the short group gets no recording selection, the source no new mask.
        n_recordings = len(RecordingSelection())
        n_artifacts = len(RecordingArtifactSelection())
        before = _row_counts()
        with pytest.raises(ValueError, match="is missing"):
            run_v2_pipeline(
                **short_inputs,
                preflight=False,
                manual_excluded_times=[[t0 + 14.0, t0 + 14.5]],
                motion_mode="estimate",
                motion_correction_params_name="no_such_recipe",
            )
        assert len(RecordingSelection()) == n_recordings
        assert len(RecordingArtifactSelection()) == n_artifacts
        assert _row_counts() == before

        # The short group's recording was never computed, so the ids that
        # fold in its content hash are pending rather than guessed.
        short = preflight_v2_pipeline(
            **short_inputs,
            motion_mode="apply",
            motion_correction_params_name=MOTION_RECIPE,
        ).expected_ids
        for name in (
            "motion_estimate_id",
            "motion_corrected_recording_id",
            "sorting_id",
        ):
            assert short[name]["id"] is None and short[name]["pending"]
        # Estimating does not sort a corrected recording, so the sorter's own
        # correction is not a motion error there.
        estimate_report = preflight_v2_pipeline(
            **{**inputs, "pipeline_preset": self_correcting_preset},
            motion_mode="estimate",
            motion_correction_params_name=MOTION_RECIPE,
        )
        assert "sorter_motion_correction_off" not in {
            c.name for c in estimate_report.checks
        }
    finally:
        monkeypatch.undo()
        _drop_artifact_detections(artifact_ids)
        recipes = [{"motion_correction_params_name": n} for n in test_recipes]
        (MotionCorrectionParameters & recipes).super_delete(
            warn=False, safemode=False
        )
        (
            MotionEstimationParameters
            & {"motion_estimation_params_name": "no_evidence_test"}
        ).super_delete(warn=False, safemode=False)


@pytest.mark.parametrize(
    "case, message",
    [
        ("self_correcting_sorter", "apply_motion_correction=False"),
        ("missing_recipe", "is missing"),
        ("short_member", "concat member 0 sort_group_id=99"),
    ],
)
def test_concat_preflight_refuses_unsupported_motion(
    discontinuous_sources,
    self_correcting_preset,
    short_sort_group,
    case,
    message,
):
    """The concat preflight applies the same motion checks -- a sorter row
    that corrects motion itself under ``apply``, a recipe with no row, a
    member whose geometry the recipe cannot estimate on -- and fails before
    any member, concat, motion or sort row is built."""
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        SessionGroup,
    )
    from tests.spikesorting.v2._ingest_helpers import configure_v2_run_inputs

    group = {
        "session_group_owner": MOTION_TEAM,
        "session_group_name": CONCAT_GROUP,
    }
    preset, recipe = PIPELINE_PRESET, MOTION_RECIPE
    if case == "self_correcting_sorter":
        preset = self_correcting_preset
    elif case == "missing_recipe":
        recipe = "no_such_recipe"
    else:
        group["session_group_name"] = "motion_short_concat"
        members = [
            {
                **configure_v2_run_inputs(
                    short_sort_group["nwb_file_name"],
                    MOTION_TEAM,
                    interval_list_name=name,
                ),
                "sort_group_id": short_sort_group["sort_group_id"],
            }
            for name in (MEMBER_A_INTERVAL, MEMBER_B_INTERVAL)
        ]
        SessionGroup.create_group(
            MOTION_TEAM, group["session_group_name"], members
        )
    before = {**_row_counts(), "concat": len(ConcatenatedRecording())}
    try:
        with pytest.raises(PreflightError, match=message):
            run_v2_pipeline(
                concat_session_group_owner=group["session_group_owner"],
                concat_session_group_name=group["session_group_name"],
                pipeline_preset=preset,
                motion_mode="apply",
                motion_correction_params_name=recipe,
            )
        assert {
            **_row_counts(),
            "concat": len(ConcatenatedRecording()),
        } == before
    finally:
        if case == "short_member":
            (SessionGroup & group).super_delete(warn=False, safemode=False)


@pytest.fixture
def unfiltered_preset(dj_conn):
    """A registered preset whose preprocessing recipe is ``no_filter``."""
    from spyglass.spikesorting.v2._pipeline_presets import (
        _PIPELINE_PRESETS,
        register_pipeline_preset,
    )

    name = "motion_test_no_filter_2026_09"
    register_pipeline_preset(
        name,
        _PIPELINE_PRESETS[PIPELINE_PRESET].model_copy(
            update={"preprocessing_params_name": "no_filter"}
        ),
    )
    yield name
    _PIPELINE_PRESETS.pop(name, None)


@pytest.mark.parametrize("mode", ["estimate", "apply"])
def test_preflight_refuses_motion_on_an_unfiltered_recipe(
    drift_recording, discontinuous_sources, unfiltered_preset, mode
):
    """Single-session and concat preflight refuse a motion mode when the
    preset's preprocessing recipe applies no temporal filter, and the run
    fails before any recording, concat, motion or sort row is built."""
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.pipeline import (
        preflight_v2_pipeline,
        run_v2_pipeline,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    message = "'no_filter' applies no temporal filter"
    motion = {
        "motion_mode": mode,
        "motion_correction_params_name": MOTION_RECIPE,
    }
    inputs = {
        **_pipeline_inputs(drift_recording),
        "pipeline_preset": unfiltered_preset,
    }

    def _counts():
        return {
            **_row_counts(),
            "recording_selections": len(RecordingSelection()),
            "concat_selections": len(ConcatenatedRecordingSelection()),
        }

    report = preflight_v2_pipeline(**inputs, **motion)
    failed = {c.name: c.fix for c in report.checks if not c.ok}
    assert list(failed) == ["motion_source_filtered"], failed
    assert message in failed["motion_source_filtered"]
    assert "filtered, unwhitened traces" in failed["motion_source_filtered"]
    # The filtering recipe of the same run passes the check.
    passing = preflight_v2_pipeline(
        **_pipeline_inputs(drift_recording), **motion
    )
    assert {c.name: c.ok for c in passing.checks}["motion_source_filtered"]

    before = _counts()
    with pytest.raises(PreflightError, match=message):
        run_v2_pipeline(**inputs, **motion)
    with pytest.raises(PreflightError, match=message):
        run_v2_pipeline(
            concat_session_group_owner=MOTION_TEAM,
            concat_session_group_name=CONCAT_GROUP,
            pipeline_preset=unfiltered_preset,
            **motion,
        )
    assert _counts() == before
