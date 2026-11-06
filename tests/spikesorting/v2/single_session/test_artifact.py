"""Artifact detection selection, populate, delete and detection tests."""

from __future__ import annotations

import numpy as np
import pytest

from tests.spikesorting.v2.single_session._helpers import (
    _build_synthetic_rec,
)


@pytest.mark.slow
def test_shared_artifact_group_insert_validates_inputs(populated_recording):
    """``SharedArtifactGroup.insert_group`` validates session
    consistency, recording existence, sampling-frequency match,
    duration / n_samples match, and non-empty members.

    Same NWB does not imply the same time axis (``interval_list_name`` is
    part of the selection), and ``si.aggregate_channels`` needs one.
    """
    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup

    # Empty members -> ValueError.
    with pytest.raises(ValueError, match="members list is empty"):
        SharedArtifactGroup.insert_group("empty_group", [])

    # Member missing recording_id -> ValueError.
    with pytest.raises(ValueError, match="must include 'recording_id'"):
        SharedArtifactGroup.insert_group(
            "bad_member", [{"not_recording_id": "x"}]
        )

    # Non-existent recording_id -> ValueError.
    import uuid

    bogus_rid = uuid.uuid4()
    with pytest.raises(ValueError, match="not in Recording"):
        SharedArtifactGroup.insert_group(
            "bogus_recording",
            [{"recording_id": bogus_rid}],
        )

    # Happy path: one member writes master + Member rows. Multi-member
    # mismatches are tested below.
    SharedArtifactGroup.insert_group(
        "v2_solo_group",
        [{"recording_id": populated_recording["recording_id"]}],
    )
    assert (
        len(
            SharedArtifactGroup
            & {"shared_artifact_group_name": "v2_solo_group"}
        )
        == 1
    )
    assert (
        len(
            SharedArtifactGroup.Member
            & {"shared_artifact_group_name": "v2_solo_group"}
        )
        == 1
    )


@pytest.mark.slow
def test_shared_artifact_group_insert_rejects_mismatched_durations(
    populated_recording, polymer_smoke_session
):
    """``insert_group`` rejects members whose recordings span
    different time windows (different ``interval_list_name`` on the
    upstream selection).

    Raises at insert time rather than as an opaque shape mismatch inside
    ``si.aggregate_channels``. Different durations trip the exact-timestamp
    check before the ``n_samples`` check.
    """
    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]

    # Second Recording on the same NWB with a shorter interval.
    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    truncated = np.asarray([[t0, t0 + 1.5]])  # 1.5s vs full ~4s
    truncated_interval = "v2_shared_group_test_truncated"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": truncated_interval,
            "valid_times": truncated,
            "pipeline": "shared_group_validation_test",
        },
        skip_duplicates=True,
    )
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    second_rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": truncated_interval,
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(second_rec_pk, reserve_jobs=False)

    with pytest.raises(ValueError, match="differing exact timestamps"):
        SharedArtifactGroup.insert_group(
            "v2_mismatched_durations",
            [
                {"recording_id": populated_recording["recording_id"]},
                {"recording_id": second_rec_pk["recording_id"]},
            ],
        )
    # No master row was created.
    assert (
        len(
            SharedArtifactGroup
            & {"shared_artifact_group_name": "v2_mismatched_durations"}
        )
        == 0
    )


@pytest.mark.slow
def test_shared_artifact_group_insert_rejects_mismatched_dtypes(
    populated_recording, polymer_smoke_session, monkeypatch
):
    """``insert_group`` rejects members whose preprocessed
    recordings have differing ``get_dtype()`` values.

    ``Recording.get_recording`` is stubbed to return recordings that differ
    only in dtype; the fixtures have no real recording pair that does.
    """
    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    second_interval = "v2_shared_group_test_dtype_window"
    # Second recording_id to group (2.5 s clears min_segment_length=1.0 s).
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": second_interval,
            "valid_times": np.asarray([[t0, t0 + 2.5]]),
            "pipeline": "shared_group_dtype_test",
        },
        skip_duplicates=True,
    )
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    second_rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": second_interval,
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(second_rec_pk, reserve_jobs=False)

    rid_a = populated_recording["recording_id"]
    rid_b = second_rec_pk["recording_id"]

    class _FakeRec:
        """SI stub with shared timestamps and a given dtype.

        Models the SI methods ``insert_group`` calls.
        """

        def __init__(self, n_samples, dtype_str, fs=30000.0):
            self._n = n_samples
            self._dtype = dtype_str
            self._fs = float(fs)
            self._times = np.arange(n_samples, dtype=np.float64) / self._fs

        def get_num_samples(self, segment_index=None):
            return self._n

        def get_sampling_frequency(self):
            return self._fs

        def get_dtype(self):
            return self._dtype

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

    def _fake_get_recording(self, key):
        # Same n_samples, so only the dtype check can raise.
        if str(key["recording_id"]) == str(rid_a):
            return _FakeRec(n_samples=30000, dtype_str="float32")
        return _FakeRec(n_samples=30000, dtype_str="int16")

    monkeypatch.setattr(Recording, "get_recording", _fake_get_recording)

    with pytest.raises(ValueError, match="differing dtypes"):
        SharedArtifactGroup.insert_group(
            "v2_mismatched_dtypes",
            [{"recording_id": rid_a}, {"recording_id": rid_b}],
        )
    assert (
        len(
            SharedArtifactGroup
            & {"shared_artifact_group_name": "v2_mismatched_dtypes"}
        )
        == 0
    )


@pytest.mark.slow
def test_artifact_detection_populates_and_writes_interval_list(
    populated_recording,
):
    """``RecordingArtifactDetection.make`` writes one ``IntervalList`` row
    under ``f"artifact_detection_{artifact_detection_id}"`` and the row is
    fetchable via ``get_artifact_removed_intervals``."""
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection

    ArtifactDetectionParameters.insert_default()
    existing = (
        RecordingArtifactSelection
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch("KEY", as_dict=True)
    for row in existing:
        # cascade: a prior test's populated_sorting may still reference it.
        (
            RecordingArtifactDetection
            & {"artifact_detection_id": row["artifact_detection_id"]}
        ).cascade_delete()
        (
            RecordingArtifactSelection
            & {"artifact_detection_id": row["artifact_detection_id"]}
        ).super_delete(warn=False)

    pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    RecordingArtifactDetection.populate(pk, reserve_jobs=False)
    assert len(RecordingArtifactDetection & pk) == 1

    nwb_file_name = (RecordingSelection & populated_recording).fetch1(
        "nwb_file_name"
    )
    interval_list_name = f"artifact_detection_{pk['artifact_detection_id']}"
    saved = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        }
    ).fetch1("valid_times")
    assert saved.shape[1] == 2  # (n_intervals, 2)
    # The "none" preset means no artifacts removed -> full window.
    assert saved.shape[0] == 1
    assert saved[0][0] < saved[0][-1]

    retrieved = RecordingArtifactDetection().get_artifact_removed_intervals(pk)
    assert (retrieved == saved).all()

    # as_dict=True returns {nwb_file_name: array} for a single recording too.
    retrieved_dict = (
        RecordingArtifactDetection().get_artifact_removed_intervals(
            pk, as_dict=True
        )
    )
    assert isinstance(retrieved_dict, dict)
    assert set(retrieved_dict) == {nwb_file_name}
    assert (retrieved_dict[nwb_file_name] == saved).all()


@pytest.mark.slow
def test_artifact_detection_delete_removes_interval_list_row(
    populated_recording,
):
    """``RecordingArtifactDetection.delete()`` cleans up the matching
    ``IntervalList`` row (DataJoint does not cascade through
    ``interval_list_name``-keyed dependencies)."""
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection

    ArtifactDetectionParameters.insert_default()
    existing = (
        RecordingArtifactSelection
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch("KEY", as_dict=True)
    for row in existing:
        # cascade: a prior test's populated_sorting may still reference it.
        (
            RecordingArtifactDetection
            & {"artifact_detection_id": row["artifact_detection_id"]}
        ).cascade_delete()
        (
            RecordingArtifactSelection
            & {"artifact_detection_id": row["artifact_detection_id"]}
        ).super_delete(warn=False)

    pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    RecordingArtifactDetection.populate(pk, reserve_jobs=False)

    nwb_file_name = (RecordingSelection & populated_recording).fetch1(
        "nwb_file_name"
    )
    interval_list_name = f"artifact_detection_{pk['artifact_detection_id']}"
    assert IntervalList & {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": interval_list_name,
    }

    (RecordingArtifactDetection & pk).delete()

    assert not (RecordingArtifactDetection & pk)
    assert not (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        }
    )


# ---------- Artifact masking at the signal level -------------------------


@pytest.mark.slow
def test_apply_artifact_mask_zeroes_artifact_frames(populated_recording):
    """``apply_artifact_mask`` actually zeros artifact frames.

    Traces are zero inside a known gap in ``valid_times``, including its
    last frame, and unchanged outside.
    """
    import numpy as np
    import uuid

    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        apply_artifact_mask,
    )

    recording = Recording().get_recording(
        {"recording_id": populated_recording["recording_id"]}
    )
    timestamps = recording.get_times()
    n_frames = len(timestamps)
    assert n_frames > 400, "fixture too short for boundary test"

    # Carve out a synthetic artifact at frames [100, 200) (half-open).
    # valid_times excludes this range: [t0, t[100]) ∪ [t[200], t[-1]].
    artifact_start_f, artifact_end_f = 100, 200
    valid_times = np.asarray(
        [
            [timestamps[0], timestamps[artifact_start_f]],
            [timestamps[artifact_end_f], timestamps[-1]],
        ]
    )

    nwb_file_name = (
        RecordingSelection
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch1("nwb_file_name")
    synthetic_artifact_id = uuid.uuid4()
    interval_list_name = f"artifact_detection_{synthetic_artifact_id}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
            "valid_times": valid_times,
            "pipeline": "spikesorting_artifact_detection_v2_test",
        }
    )
    try:
        masked = apply_artifact_mask(
            recording=recording,
            valid_times=valid_times,
        )

        gap = masked.get_traces(
            start_frame=artifact_start_f, end_frame=artifact_end_f
        )
        assert np.all(gap == 0), (
            "Artifact frames [100, 200) are not all zero; "
            f"max abs value in gap = {np.abs(gap).max()}."
        )

        # First valid frame after the gap is unmasked; check the source is
        # non-zero so the comparison is not vacuous.
        unmasked_boundary = recording.get_traces(
            start_frame=artifact_end_f, end_frame=artifact_end_f + 1
        )
        assert np.abs(unmasked_boundary).max() > 0, (
            f"source sample at boundary frame {artifact_end_f} is all-zero; "
            "the masked-vs-unmasked comparison below would be vacuous -- pick "
            "a boundary frame whose preprocessed signal is non-zero"
        )
        masked_boundary = masked.get_traces(
            start_frame=artifact_end_f, end_frame=artifact_end_f + 1
        )
        np.testing.assert_array_equal(
            masked_boundary,
            unmasked_boundary,
            err_msg=(
                f"Frame {artifact_end_f} (first valid frame after the "
                "artifact gap) differs between masked and unmasked "
                "recordings -- off-by-one boundary regression."
            ),
        )

        # Last artifact frame is masked.
        last_artifact = masked.get_traces(
            start_frame=artifact_end_f - 1, end_frame=artifact_end_f
        )
        assert np.all(last_artifact == 0), (
            f"Last artifact frame {artifact_end_f - 1} not zeroed -- "
            "this is the off-by-one boundary the v2 fix corrected."
        )
    finally:
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_list_name,
            }
        ).super_delete(warn=False)


# ---------- ArtifactDetection: signal-level detect_artifacts ------------


def test_detect_artifacts_finds_known_transient(dj_conn):
    """``detect_artifacts`` excludes a planted amplitude transient.

    A 200 uV transient at frames 1000-1199 yields two valid intervals around
    it. ``dj_conn`` is needed only because importing the artifact module
    declares its schema.
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    # Gain 1.0, so traces are in uV.
    fs = 30_000.0
    n_samples = 5000
    n_channels = 4
    traces = np.zeros((n_samples, n_channels), dtype=np.float32)
    traces[1000:1200, :] = 200.0
    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
    rec.set_channel_gains([1.0] * n_channels)

    # removal_window_ms must be > 0; 0.05 ms widens the span by
    # ceil(0.05 * 30000 / 2000) = 1 frame per side.
    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=0.05,
        join_window_ms=0.0,
        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
    )

    valid_times = detect_artifacts(rec, params)
    timestamps = rec.get_times()

    # Widened span: [999, 1200].
    expected_start_f = 999  # 1000 - half_window
    expected_end_f = 1200  # 1199 + half_window
    expected_end_open_f = expected_end_f + 1  # half-open boundary

    assert valid_times.shape == (2, 2), (
        f"Expected exactly 2 valid intervals around the synthetic "
        f"transient at frames 1000-1199, got shape {valid_times.shape}. "
        "detect_artifacts either skipped the detection body or "
        "merged the artifact and valid runs incorrectly."
    )
    assert valid_times[0][0] == pytest.approx(timestamps[0], abs=1e-9)
    assert valid_times[0][1] == pytest.approx(
        timestamps[expected_start_f], abs=1e-9
    ), (
        f"First valid interval ends at {valid_times[0][1]}, expected "
        f"{timestamps[expected_start_f]} (start of the expanded "
        "artifact span)."
    )
    # Second interval starts after the last artifact frame.
    assert valid_times[1][0] == pytest.approx(
        timestamps[expected_end_open_f], abs=1e-9
    ), (
        f"Second valid interval starts at {valid_times[1][0]}, expected "
        f"{timestamps[expected_end_open_f]} (first sample AFTER the "
        "expanded artifact span). If this is "
        f"``timestamps[{expected_end_f}]`` the off-by-one fix at "
        "the END of the artifact interval has regressed."
    )
    assert valid_times[1][1] == pytest.approx(timestamps[-1], abs=1e-9)


def test_detect_artifacts_no_threshold_crossings(dj_conn):
    """``detect_artifacts`` returns the full recording window as a
    single valid interval when no frame trips the threshold.

    Detection runs (``detect=True``) and finds nothing.
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    # All-zero traces.
    rec = si.NumpyRecording(
        traces_list=[np.zeros((2000, 4), dtype=np.float32)],
        sampling_frequency=30_000.0,
    )
    rec.set_channel_gains([1.0] * 4)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=10.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
    )
    valid_times = detect_artifacts(rec, params)
    timestamps = rec.get_times()

    assert valid_times.shape == (1, 2)
    assert valid_times[0][0] == pytest.approx(timestamps[0], abs=1e-9)
    assert valid_times[0][1] == pytest.approx(timestamps[-1], abs=1e-9)


def test_detect_artifacts_zscore_only_detection(dj_conn):
    """``detect_artifacts`` runs the z-score-only branch when
    ``amplitude_threshold_uv is None``.

    The z-score is across channels per frame (as in v1), so a 10-sample
    excursion on one channel trips it.
    """
    import numpy as np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    rng = np.random.default_rng(42)
    # One channel's cross-channel z is at most ~sqrt(N-1): ~5.6 for 32
    # channels, so a 4.0 threshold is reachable (4 channels cap at ~1.7).
    n_samples, n_channels = 5000, 32
    traces = rng.normal(0.0, 0.5, size=(n_samples, n_channels)).astype(
        np.float32
    )
    traces[2000:2010, 0] = 500.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=None,  # z-score only
        zscore_threshold=4.0,
        proportion_above_threshold=1.0 / n_channels,  # any-1-channel triggers
        removal_window_ms=0.05,
        join_window_ms=0.0,
        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
    )
    valid_times = detect_artifacts(rec, params)
    timestamps = rec.get_times()

    assert valid_times.shape == (2, 2), (
        f"z-score-only detection expected 2 valid intervals around the "
        f"transient at frames 2000-2009, got shape {valid_times.shape}."
    )
    # Pinpoint the artifact run via the expanded boundaries
    # (half_window_frames=1).
    assert valid_times[0][1] == pytest.approx(timestamps[1999], abs=1e-9)
    assert valid_times[1][0] == pytest.approx(timestamps[2011], abs=1e-9)


def test_detect_artifacts_amplitude_and_zscore_combined(dj_conn):
    """``detect_artifacts`` OR-combines the amplitude and z-score detectors.

    As in v1, a channel is flagged when either detector trips. Each detector
    gets a region only it can see:

    - frames 1000-1099: every channel at 100 uV (amplitude only);
    - frames 3000-3099: channel 0 alone at 20 uV (z-score only).

    Both regions are excised. Background and thresholds match
    ``test_detect_artifacts_zscore_only_detection``.
    """
    import numpy as np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    rng = np.random.default_rng(42)
    n_samples, n_channels = 5000, 32
    traces = rng.normal(0.0, 0.5, size=(n_samples, n_channels)).astype(
        np.float32
    )
    # Amplitude only: uniform across channels, so cross-channel z ~ 0.
    traces[1000:1100, :] = 100.0
    # Z-score only: below the 50 uV amplitude threshold.
    traces[3000:3100, 0] = 20.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=4.0,
        proportion_above_threshold=1.0 / n_channels,  # any 1 channel
        removal_window_ms=0.05,
        join_window_ms=0.0,
        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
    )
    valid_times = detect_artifacts(rec, params)
    timestamps = rec.get_times()

    def _covered(t):
        return any(lo <= t <= hi for lo, hi in valid_times)

    # Both regions excised; quiet background stays valid.
    assert not _covered(timestamps[1050]), (
        "amplitude-only region (frames 1000-1099) is still valid -- the "
        "amplitude branch did not contribute, or the combine became AND"
    )
    assert not _covered(timestamps[3050]), (
        "z-score-only region (frames 3000-3099) is still valid -- the "
        "z-score branch did not contribute, or the combine became AND"
    )
    assert _covered(
        timestamps[500]
    ), "a quiet background frame was flagged; detection is over-triggering"
    assert valid_times.shape == (3, 2), (
        "expected 3 valid intervals around the two single-detector regions, "
        f"got shape {valid_times.shape}"
    )


def test_detect_artifacts_join_window_merges_runs(dj_conn):
    """``detect_artifacts`` merges two artifact runs separated by
    fewer than ``join_window_frames`` into a single artifact span.

    Two transients 10 frames apart merge with a 30-frame join window and
    stay separate with a 3-frame one.
    """
    import numpy as np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    n_samples, n_channels = 5000, 4
    traces = np.zeros((n_samples, n_channels), dtype=np.float32)
    traces[1000:1100, :] = 100.0  # artifact A
    traces[1110:1210, :] = 100.0  # artifact B; 10 frames after A
    rec = _build_synthetic_rec(traces)

    # join_window_ms = 1.0 -> join_window_frames = ceil(1 * 30 / 1) =
    # 30 frames. Gap between A and B is 10 frames, so they merge.
    merged = detect_artifacts(
        rec,
        ArtifactDetectionParamsSchema(
            detect=True,
            amplitude_threshold_uv=50.0,
            zscore_threshold=None,
            proportion_above_threshold=0.5,
            removal_window_ms=0.05,
            join_window_ms=1.0,
            min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
        ),
    )
    assert merged.shape == (2, 2), (
        f"With join_window_ms=1.0 (= 30 frames > 10-frame gap), the two "
        f"transients should merge into one artifact span, leaving 2 "
        f"valid intervals; got shape {merged.shape}."
    )

    # join_window_ms = 0.1 -> 3 frames; the 10-frame gap stays.
    # min_length_s below the ~0.27 ms middle interval (8 frames) keeps it.
    unmerged = detect_artifacts(
        rec,
        ArtifactDetectionParamsSchema(
            detect=True,
            amplitude_threshold_uv=50.0,
            zscore_threshold=None,
            proportion_above_threshold=0.5,
            removal_window_ms=0.05,
            join_window_ms=0.1,
            min_length_s=0.0001,
        ),
    )
    assert unmerged.shape == (3, 2), (
        f"With join_window_ms=0.1 (= 3 frames < 10-frame gap), the two "
        f"transients stay separate, leaving 3 valid intervals; got "
        f"shape {unmerged.shape}."
    )


@pytest.mark.slow
def test_artifact_detection_parameters_validates_via_insert1(dj_conn):
    """``ArtifactDetectionParameters.insert1`` rejects an unknown params key."""
    from pydantic import ValidationError

    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
    )

    bad_row = {
        "artifact_detection_params_name": "params_validates_unit_test",
        "params": {
            "detect": True,
            "amplitude_threshold_uv": 100.0,
            "not_a_real_field": "x",  # rejected by extra="forbid"
        },
        "params_schema_version": 1,
        "job_kwargs": None,
    }
    with pytest.raises(ValidationError):
        ArtifactDetectionParameters().insert1(bad_row)

    assert (
        len(
            ArtifactDetectionParameters
            & {"artifact_detection_params_name": "params_validates_unit_test"}
        )
        == 0
    ), "Failed validation should not have written a row."


# ---------- detect_artifacts cross-channel proportion boundary ----------


def test_detect_artifacts_below_proportion_threshold_ignored(dj_conn):
    """``detect_artifacts`` does NOT flag artifact frames when
    fewer than ``proportion_above_threshold`` of channels exceed the
    amplitude threshold.

    A 200 uV transient on one of 4 channels does not meet
    ``proportion_above_threshold=0.5`` (2 channels).
    """
    import numpy as np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    n_samples, n_channels = 5000, 4
    traces = np.zeros((n_samples, n_channels), dtype=np.float32)
    traces[1000:1100, 0] = 200.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=0.05,
        join_window_ms=0.0,
        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
    )
    valid_times = detect_artifacts(rec, params)
    timestamps = rec.get_times()

    assert valid_times.shape == (1, 2), (
        f"Single-channel transient should NOT trip cross-channel "
        f"proportion gate (1/4 < 0.5 required); expected one full "
        f"valid interval but got shape {valid_times.shape}."
    )
    assert valid_times[0][0] == pytest.approx(timestamps[0], abs=1e-9)
    assert valid_times[0][1] == pytest.approx(timestamps[-1], abs=1e-9)


# ---------- ArtifactDetection.delete ownership rows -----------------------


@pytest.mark.slow
def test_artifact_detection_delete_uses_interval_ownership_part_rows(
    populated_recording,
):
    """``RecordingArtifactDetection.delete`` cleans via owned part rows.

    Ownership comes from ``RemovedInterval`` rows, not interval names.
    """
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )

    ArtifactDetectionParameters.insert_default()
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    # Start from a detection no sort references (a prior test's
    # populated_sorting may; that fixture rebuilds its chain on demand).
    (RecordingArtifactDetection & art_pk).cascade_delete(safemode=False)
    RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
    part_rows = (RecordingArtifactDetection.RemovedInterval & art_pk).fetch(
        "nwb_file_name", "interval_list_name", as_dict=True
    )
    assert (
        part_rows
    ), "RecordingArtifactDetection.populate must insert ownership rows"

    (RecordingArtifactDetection & art_pk).delete(safemode=False)
    assert len(RecordingArtifactDetection & art_pk) == 0
    assert len(IntervalList & part_rows) == 0


@pytest.mark.slow
def test_artifact_detection_delete_requires_interval_ownership_part_rows(
    populated_recording,
):
    """A row without ownership parts is invalid and must fail loudly.

    Ownership is not reconstructed from interval names, which would expose
    the rows to ``IntervalList.cleanup`` as orphans.
    """
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )

    ArtifactDetectionParameters.insert_default()
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    # Start from a detection no sort references (a prior test's
    # populated_sorting may; that fixture rebuilds its chain on demand).
    (RecordingArtifactDetection & art_pk).cascade_delete(safemode=False)
    RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
    part_rows = (RecordingArtifactDetection.RemovedInterval & art_pk).fetch(
        "nwb_file_name", "interval_list_name", as_dict=True
    )
    assert (
        part_rows
    ), "RecordingArtifactDetection.populate must insert ownership rows"

    (RecordingArtifactDetection.RemovedInterval & art_pk).delete_quick()

    try:
        with pytest.raises(ValueError, match="no RemovedInterval part rows"):
            (RecordingArtifactDetection & art_pk).delete(safemode=False)
        assert len(RecordingArtifactDetection & art_pk) == 1
    finally:
        # force_masters: the detection is also an ArtifactDetectionOutput part.
        if RecordingArtifactDetection & art_pk:
            (RecordingArtifactDetection & art_pk).super_delete(
                warn=False, force_masters=True
            )
        (IntervalList & part_rows).delete_quick()


# ---------- artifact at recording boundary boundary clamp ----------------


def test_detect_artifacts_clamps_artifact_at_recording_end(dj_conn):
    """``detect_artifacts`` clamps the half-open end of an artifact
    that runs to the last sample.

    An artifact reaching the last sample does not index past the end.
    """
    import numpy as np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts

    n_samples, n_channels = 1000, 4
    traces = np.zeros((n_samples, n_channels), dtype=np.float32)
    # Transient at the very end of the recording.
    traces[990:1000, :] = 200.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=0.05,
        join_window_ms=0.0,
        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
    )
    valid_times = detect_artifacts(rec, params)
    timestamps = rec.get_times()

    # Artifact runs from frame 990 to the end: one valid interval before it.
    assert valid_times.shape == (1, 2), (
        f"Expected one valid interval ending before the boundary "
        f"artifact, got shape {valid_times.shape}."
    )
    assert valid_times[0][0] == pytest.approx(timestamps[0], abs=1e-9)
    assert valid_times[0][1] < timestamps[990] + 1e-9, (
        f"Valid interval ends at {valid_times[0][1]}, after the "
        f"start of the boundary artifact ({timestamps[990]}). "
        "The boundary clamp may have an off-by-one regression."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_shared_group_make_compute_reads_fetched_members_without_db(
    populated_recording, monkeypatch
):
    """``make_fetch`` resolves each member's cached file (hash-stable across
    DataJoint's two fetches) and ``make_compute`` scans the members with no
    DB access, matching a scan of ``Recording().get_recording``."""
    import numpy as np

    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        SharedArtifactGroup,
        SharedGroupArtifactDetection,
        SharedGroupArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from tests.spikesorting.v2._tripart_helpers import (
        fetch_hash,
        forbid_db_queries,
    )

    ArtifactDetectionParameters.insert_default()
    group_name = "v2_shared_group_compute_guard"
    group = SharedArtifactGroup & {"shared_artifact_group_name": group_name}
    group.super_delete(warn=False)
    rec_key = {"recording_id": populated_recording["recording_id"]}
    try:
        SharedArtifactGroup.insert_group(group_name, [rec_key])
        art_pk = SharedGroupArtifactSelection.insert_selection(
            {
                "shared_artifact_group_name": group_name,
                "artifact_detection_params_name": "default",
            }
        )
        table = SharedGroupArtifactDetection()
        fetched = table.make_fetch(art_pk)
        assert fetched.member_traces == (
            Recording().resolve_stored_traces(rec_key),
        )
        assert fetch_hash(table.make_fetch(art_pk)) == fetch_hash(fetched)
        with forbid_db_queries(monkeypatch, "SharedGroupArtifactDetection"):
            computed = table.make_compute(art_pk, *fetched)

        expected = table._run_artifact_scan(
            Recording().get_recording(rec_key),
            fetched.validated,
            fetched.artifact_job_kwargs,
            context="",
            manual_excluded_times=fetched.manual_excluded_times,
        )
        np.testing.assert_array_equal(computed.valid_times, expected)
    finally:
        group.super_delete(warn=False)


@pytest.mark.slow
@pytest.mark.integration
def test_shared_artifact_group_populate_end_to_end(
    populated_recording, polymer_smoke_session
):
    """End-to-end shared-group populate writes one IntervalList per
    member ``nwb_file_name``.

    The smoke fixture gives a one-member group, so one row is written.
    """
    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        SharedArtifactGroup,
        SharedGroupArtifactDetection,
        SharedGroupArtifactSelection,
    )
    from spyglass.spikesorting.v2._artifacts.naming import (
        artifact_detection_interval_list_name,
    )

    ArtifactDetectionParameters.insert_default()
    group_name = "v2_e2e_shared_group"

    (
        SharedArtifactGroup & {"shared_artifact_group_name": group_name}
    ).super_delete(warn=False)

    SharedArtifactGroup.insert_group(
        group_name,
        [{"recording_id": populated_recording["recording_id"]}],
    )

    art_pk = SharedGroupArtifactSelection.insert_selection(
        {
            "shared_artifact_group_name": group_name,
            "artifact_detection_params_name": "none",
        }
    )
    # ``none`` preset: full-window valid interval, independent of amplitude.
    SharedGroupArtifactDetection.populate(art_pk, reserve_jobs=False)

    interval_name = artifact_detection_interval_list_name(
        art_pk["artifact_detection_id"]
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    rows = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }
    ).fetch(as_dict=True)
    assert len(rows) == 1
    assert rows[0]["pipeline"] == "spikesorting_artifact_detection_v2"
    assert SharedGroupArtifactDetection.RemovedInterval & {
        "artifact_detection_id": art_pk["artifact_detection_id"],
        "nwb_file_name": nwb_file_name,
        "interval_list_name": interval_name,
    }, "SharedGroupArtifactDetection.populate did not insert the ownership part row"

    # Shared-group sources return {member nwb_file_name: valid_times}.
    import numpy as np

    shared_intervals = (
        SharedGroupArtifactDetection().get_artifact_removed_intervals(art_pk)
    )
    assert isinstance(shared_intervals, dict), (
        f"get_artifact_removed_intervals returned {type(shared_intervals)}; "
        "expected dict for shared-group source."
    )
    assert nwb_file_name in shared_intervals
    np.testing.assert_array_equal(
        shared_intervals[nwb_file_name], rows[0]["valid_times"]
    )

    # delete() removes the IntervalList rows of every member.
    (SharedGroupArtifactDetection & art_pk).delete(safemode=False)
    remaining = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }
    ).fetch(as_dict=True)
    assert len(remaining) == 0, (
        "SharedGroupArtifactDetection.delete() left an orphan IntervalList "
        "row for a shared-group artifact; the delete-cleanup "
        "branch for shared sources is broken."
    )


# ---------- error-handling context in warnings and reads -----------------


def test_artifact_empty_warning_has_context(dj_conn, monkeypatch):
    """The zero-artifact-frames warning names the artifact_detection_id and the
    active thresholds, not just a bare 'zero frames' message.
    """
    import numpy as np

    from spyglass.spikesorting.v2._artifacts.intervals import detect_artifacts
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.utils import logger

    captured = []
    monkeypatch.setattr(
        logger,
        "warning",
        lambda msg, *a, **k: captured.append(msg),
    )

    rec = _build_synthetic_rec(np.zeros((1000, 4), dtype=np.float32))
    validated = ArtifactDetectionParamsSchema(
        detect=True, amplitude_threshold_uv=500.0
    )
    out = detect_artifacts(
        rec, validated, context=" for artifact_detection_id=abc-123"
    )
    assert out.shape == (1, 2)  # full window returned on zero artifacts
    assert any("abc-123" in m for m in captured), captured
    assert any("amplitude_threshold_uv" in m for m in captured)


@pytest.mark.slow
@pytest.mark.integration
def test_shared_artifact_group_multi_member_union(
    populated_recording, polymer_smoke_session, monkeypatch
):
    """The union scan reads both members and applies its channel quorum.

    Two time-aligned members each get a transient. With
    ``proportion_above_threshold=0.4`` over the 8-channel union, a 4-channel
    transient is removed and a 3-channel excursion is not (a per-member
    denominator would flag it). Every member resolves to one shared
    ``IntervalList`` row. Traces are synthetic (via monkeypatch); DB rows
    are real.
    """
    import spikeinterface as si

    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        SharedArtifactGroup,
        SharedGroupArtifactDetection,
        SharedGroupArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2._artifacts.naming import (
        artifact_detection_interval_list_name,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    fs = 30000.0
    n_samples = 90000  # 3 s
    n_ch = 4
    art_lo, art_hi = 45000, 45050  # transient at 1.5 s
    second_art_lo, second_art_hi = 15000, 15050  # member B at 0.5 s
    below_quorum_lo, below_quorum_hi = 65000, 65050

    def _synth_recording(with_artifact):
        traces = np.zeros((n_samples, n_ch), dtype=np.int16)
        if with_artifact:
            traces[art_lo:art_hi, :] = 5000
            traces[below_quorum_lo:below_quorum_hi, :3] = 5000
        else:
            traces[second_art_lo:second_art_hi, :] = 5000
        rec = si.NumpyRecording([traces], sampling_frequency=fs)
        rec.set_channel_gains([1.0] * n_ch)  # µV == counts
        rec.set_channel_offsets([0.0] * n_ch)
        return rec

    # Second recording_id in the same session; its traces are replaced below.
    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    second_interval = "v2_shared_union_window"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": second_interval,
            "valid_times": np.asarray([[t0, t0 + 2.5]]),
            "pipeline": "shared_group_union_test",
        },
        skip_duplicates=True,
    )
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    second_rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": second_interval,
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(second_rec_pk, reserve_jobs=False)

    rid_a = populated_recording["recording_id"]  # member A: artifact
    rid_b = second_rec_pk["recording_id"]  # member B: separate artifact

    def _fake_get_recording(self, key):
        return _synth_recording(
            with_artifact=str(key["recording_id"]) == str(rid_a)
        )

    # ``insert_group`` validates the members through ``get_recording``; the
    # detection reads the files its ``make_fetch`` resolved.
    from spyglass.spikesorting.v2._storage import nwb as _recording_nwb

    path_a = Recording().resolve_stored_traces({"recording_id": rid_a}).abs_path
    monkeypatch.setattr(Recording, "get_recording", _fake_get_recording)
    monkeypatch.setattr(
        _recording_nwb,
        "read_stored_traces",
        lambda traces: _synth_recording(
            with_artifact=traces.abs_path == path_a
        ),
    )

    # 4 of 8 union channels flag a frame; min_length 0.1 s keeps both halves.
    params_name = "multi_member_union_test"
    ArtifactDetectionParameters.insert1(
        {
            "artifact_detection_params_name": params_name,
            "params": ArtifactDetectionParamsSchema(
                detect=True,
                amplitude_threshold_uv=1000.0,
                zscore_threshold=None,
                proportion_above_threshold=0.4,
                removal_window_ms=1.0,
                join_window_ms=1.0,
                min_length_s=0.1,
            ).model_dump(),
            "params_schema_version": 2,
        },
        skip_duplicates=True,
    )

    group_name = "v2_multi_member_union_group"

    def _clear_shared_group():
        # Delete detections and selections before the group they reference.
        stale_art_ids = (
            SharedGroupArtifactSelection
            & {"shared_artifact_group_name": group_name}
        ).fetch("artifact_detection_id")
        for aid in stale_art_ids:
            (
                SharedGroupArtifactDetection & {"artifact_detection_id": aid}
            ).delete(safemode=False)
            (
                SharedGroupArtifactSelection & {"artifact_detection_id": aid}
            ).super_delete(warn=False)
        (
            SharedArtifactGroup & {"shared_artifact_group_name": group_name}
        ).super_delete(warn=False)

    _clear_shared_group()
    try:
        SharedArtifactGroup.insert_group(
            group_name,
            [{"recording_id": rid_a}, {"recording_id": rid_b}],
        )
        assert (
            len(
                SharedArtifactGroup.Member
                & {"shared_artifact_group_name": group_name}
            )
            == 2
        )

        art_pk = SharedGroupArtifactSelection.insert_selection(
            {
                "shared_artifact_group_name": group_name,
                "artifact_detection_params_name": params_name,
            }
        )
        SharedGroupArtifactDetection.populate(art_pk, reserve_jobs=False)

        interval_name = artifact_detection_interval_list_name(
            art_pk["artifact_detection_id"]
        )
        rows = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_name,
            }
        ).fetch(as_dict=True)
        # Single session -> one shared row covering every member.
        assert (
            len(rows) == 1
        ), f"expected one shared IntervalList row; got {len(rows)}"
        valid_times = rows[0]["valid_times"]

        # Both members contribute a different artifact window; dropping either
        # member would leave only two valid intervals.
        art_mid = 0.5 * (art_lo + art_hi) / fs  # ~1.5 s
        second_art_mid = 0.5 * (second_art_lo + second_art_hi) / fs
        assert valid_times.shape == (3, 2), (
            f"union scan should split into 3 intervals at both artifacts; got "
            f"valid_times={valid_times.tolist()} (shape {valid_times.shape}). "
            "Every member's channels must contribute."
        )
        for start, end in valid_times:
            for planted_mid in (art_mid, second_art_mid):
                assert not (start < planted_mid < end), (
                    f"a valid interval [{start}, {end}] spans the artifact at "
                    f"{planted_mid}s; the union scan did not remove it."
                )
        # Coverage survives on both sides of the artifact.
        assert any(s <= 0.2 <= e for s, e in valid_times), "pre-artifact lost"
        assert any(s <= 2.4 <= e for s, e in valid_times), "post-artifact lost"
        below_quorum_mid = 0.5 * (below_quorum_lo + below_quorum_hi) / fs
        assert any(
            s <= below_quorum_mid <= e for s, e in valid_times
        ), "three channels must remain valid under the four-of-eight quorum"

        # Every member resolves to the same artifact-removed times.
        shared = SharedGroupArtifactDetection().get_artifact_removed_intervals(
            art_pk
        )
        assert isinstance(shared, dict) and nwb_file_name in shared
        np.testing.assert_array_equal(shared[nwb_file_name], valid_times)
    finally:
        _clear_shared_group()
        (
            ArtifactDetectionParameters
            & {"artifact_detection_params_name": params_name}
        ).super_delete(warn=False, safemode=False)


# ---------- a raw NaN makes populate raise instead of inserting empty ------


@pytest.mark.slow
def test_recording_artifact_detection_raises_on_raw_nan(
    dj_conn, tmp_path, smoke_nwb
):
    """One raw NaN sample makes ``RecordingArtifactDetection.populate`` raise.

    The NaN is written into a copy of the raw fixture before ingestion.
    ``Recording.populate`` stores it; the artifact scan raises rather than
    reporting no artifacts. The bandpass filter spreads the NaN over the
    whole channel, so the first chunk raises and names only that channel.
    """
    import shutil
    from pathlib import Path

    from spyglass.common.common_lab import LabTeam
    from spyglass.settings import raw_dir
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        copy_and_insert_nwb,
        inject_raw_nan,
    )

    source = (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "mearec_polymer_smoke.nwb"
    )
    if not source.exists():
        pytest.skip(f"Fixture {source.name} not found.")

    corrupted = tmp_path / "mearec_polymer_smoke_raw_nan.nwb"
    shutil.copy2(source, corrupted)
    # Channel 0 is in the lowest sort group; frame 45_000 is mid-file.
    inject_raw_nan(corrupted, frame_index=45_000, channel_index=0)

    # Discard any stale copy; copy_and_insert_nwb would otherwise reuse it.
    dest_name = "mearec_polymer_smoke_raw_nan.nwb"
    Path(raw_dir, dest_name).unlink(missing_ok=True)
    Path(raw_dir, get_nwb_copy_filename(dest_name)).unlink(missing_ok=True)
    nwb_file_name = copy_and_insert_nwb(corrupted, dest_name=dest_name)
    session_key = {"nwb_file_name": nwb_file_name}
    try:
        PreprocessingParameters.insert_default()
        ArtifactDetectionParameters.insert_default()
        LabTeam.insert1(
            {
                "team_name": "v2_test_team",
                "team_description": "v2 pipeline tests",
            },
            skip_duplicates=True,
        )
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
        sort_group_id = int(
            sorted((SortGroupV2 & session_key).fetch("sort_group_id"))[0]
        )
        rec_pk = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "v2_test_team",
            }
        )
        # Recording does not check finiteness, so this succeeds.
        Recording.populate(rec_pk, reserve_jobs=False)
        assert Recording & rec_pk, "Recording.populate must have inserted a row"

        art_pk = RecordingArtifactSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "artifact_detection_params_name": "default",
            }
        )
        with pytest.raises(
            ValueError,
            match=(
                r"non-finite \(NaN/Inf\) samples in segment \d+ frames "
                r"\[\d+, \d+\): channel 0: \d+"
            ),
        ):
            RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)

        assert not (RecordingArtifactDetection & art_pk), (
            "a raised populate must not leave a RecordingArtifactDetection "
            "row behind"
        )
        assert not (RecordingArtifactDetection.RemovedInterval & art_pk), (
            "a raised populate must not leave an artifact interval list "
            "behind"
        )
    finally:
        _clean_session_v2(session_key)
        Path(raw_dir, dest_name).unlink(missing_ok=True)
        Path(raw_dir, get_nwb_copy_filename(dest_name)).unlink(missing_ok=True)
