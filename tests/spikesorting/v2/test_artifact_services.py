"""Tests for the ``_artifact_intervals`` artifact detect / scan / row helpers.

Covers the pure detection functions (``scan_artifact_frames`` /
``detect_artifacts``), the interval-row builders
(``build_artifact_interval_rows`` / ``build_artifact_interval_part_rows``),
and the validation-order pin on ``read_artifact_removed_intervals``.

Most tests here are DB-free (numpy / SpikeInterface only). The single
exception is the ``read_artifact_removed_intervals`` validation-order pin,
which must import the ``artifact`` schema module and therefore needs the DB
session (``dj_conn`` fixture).
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest


def _rec(traces, fs=1000.0, gain=1.0, times=None):
    import spikeinterface as si

    n_ch = traces.shape[1]
    rec = si.NumpyRecording(
        traces_list=[traces.astype("float32")], sampling_frequency=fs
    )
    rec.set_channel_gains([gain] * n_ch)
    rec.set_channel_offsets([0.0] * n_ch)
    if times is not None:
        rec.set_times(np.asarray(times, dtype=float))
    return rec


def _artifact_params(**overrides):
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )

    # removal_window_ms / min_length_s are gt=0 in the schema; keep them
    # small so the 1-frame transient widens by ~1 frame and the ~0.049 s
    # valid intervals on a 0.1 s recording survive the min-length filter.
    base = dict(
        detect=True,
        amplitude_threshold_uv=1000.0,
        zscore_threshold=None,
        proportion_above_threshold=1.0,
        removal_window_ms=0.5,
        join_window_ms=0.0,
        min_length_s=0.001,
    )
    base.update(overrides)
    return ArtifactDetectionParamsSchema.model_validate(base)


def test_detect_artifacts_detect_false_returns_full_window():
    """``detect=False`` returns the recorded window untouched (no DB)."""
    from spyglass.spikesorting.v2._artifact_intervals import detect_artifacts

    traces = np.zeros((100, 2), dtype="float32")
    rec = _rec(traces)
    vt = detect_artifacts(rec, _artifact_params(detect=False))
    assert vt.shape == (1, 2)
    assert vt[0, 0] == pytest.approx(0.0)
    # One interval spanning the recording.
    assert vt[0, 1] >= 0.099 - 1e-9


def test_scan_artifact_frames_flags_transient_only():
    """``scan_artifact_frames`` flags the artifact frames and nothing else."""
    from spyglass.spikesorting.v2._artifact_intervals import (
        scan_artifact_frames,
    )

    traces = np.zeros((100, 2), dtype="float32")
    traces[50:52, :] = 5000.0  # large transient on every channel
    rec = _rec(traces)
    flagged = scan_artifact_frames(rec, _artifact_params())
    assert set(int(f) for f in flagged) == {50, 51}


def test_detect_artifacts_excludes_transient_window():
    """``detect_artifacts`` carves the artifact out of the valid times."""
    from spyglass.spikesorting.v2._artifact_intervals import detect_artifacts

    traces = np.zeros((100, 2), dtype="float32")
    traces[50:52, :] = 5000.0
    rec = _rec(traces)
    vt = detect_artifacts(rec, _artifact_params())
    # The artifact instant (~0.050 s) is in no returned valid interval, but
    # quiet instants near the ends are.
    art_t = 50 / 1000.0
    assert not any(lo <= art_t < hi for lo, hi in vt)
    assert any(lo <= 0.0 <= hi for lo, hi in vt)
    assert any(lo <= 0.090 <= hi for lo, hi in vt)


def test_build_artifact_interval_rows_single_recording_fallback():
    """Empty ``per_member`` -> one row keyed by the master nwb_file_name."""
    from spyglass.spikesorting.v2._artifact_intervals import (
        build_artifact_interval_rows,
    )

    art_id = uuid.UUID("00000000-0000-0000-0000-0000000000aa")
    valid_times = np.array([[0.0, 1.0], [2.0, 3.0]])
    rows = build_artifact_interval_rows(
        {"artifact_detection_id": art_id},
        valid_times,
        nwb_file_name="sess.nwb",
        per_member_nwb_files=(),
    )
    assert len(rows) == 1
    (row,) = rows
    assert row["nwb_file_name"] == "sess.nwb"
    assert row["interval_list_name"] == f"artifact_detection_{art_id}"
    assert row["pipeline"] == "spikesorting_artifact_detection_v2"
    np.testing.assert_array_equal(row["valid_times"], valid_times)


def test_build_artifact_interval_rows_one_row_per_member():
    """A shared-group source writes one row per distinct member nwb file."""
    from spyglass.spikesorting.v2._artifact_intervals import (
        build_artifact_interval_rows,
    )

    art_id = uuid.UUID("00000000-0000-0000-0000-0000000000bb")
    valid_times = np.array([[0.0, 1.0]])
    rows = build_artifact_interval_rows(
        {"artifact_detection_id": art_id},
        valid_times,
        nwb_file_name="a.nwb",
        per_member_nwb_files=("a.nwb", "b.nwb"),
    )
    assert [r["nwb_file_name"] for r in rows] == ["a.nwb", "b.nwb"]
    # Same interval name + valid_times across members.
    assert {r["interval_list_name"] for r in rows} == {
        f"artifact_detection_{art_id}"
    }
    for r in rows:
        np.testing.assert_array_equal(r["valid_times"], valid_times)


def test_build_artifact_interval_part_rows_owns_interval_rows():
    """Ownership rows carry the ArtifactDetection PK + IntervalList PK only."""
    from spyglass.spikesorting.v2._artifact_intervals import (
        build_artifact_interval_part_rows,
    )

    art_id = uuid.UUID("00000000-0000-0000-0000-0000000000bc")
    interval_rows = [
        {
            "nwb_file_name": "a.nwb",
            "interval_list_name": f"artifact_detection_{art_id}",
            "valid_times": np.array([[0.0, 1.0]]),
            "pipeline": "spikesorting_artifact_detection_v2",
        },
        {
            "nwb_file_name": "b.nwb",
            "interval_list_name": f"artifact_detection_{art_id}",
            "valid_times": np.array([[0.0, 1.0]]),
            "pipeline": "spikesorting_artifact_detection_v2",
        },
    ]

    assert build_artifact_interval_part_rows(
        {"artifact_detection_id": art_id}, interval_rows
    ) == [
        {
            "artifact_detection_id": art_id,
            "nwb_file_name": "a.nwb",
            "interval_list_name": f"artifact_detection_{art_id}",
        },
        {
            "artifact_detection_id": art_id,
            "nwb_file_name": "b.nwb",
            "interval_list_name": f"artifact_detection_{art_id}",
        },
    ]


@pytest.mark.usefixtures("dj_conn")
def test_read_artifact_removed_intervals_missing_artifact_detection_id_raises_value_error():
    """Missing ``artifact_detection_id`` raises a clear ValueError before source-resolve.

    The guard runs BEFORE ``ArtifactDetectionSelection.resolve_source(key)``, so a key
    with no ``artifact_detection_id`` surfaces this targeted message rather than a
    source-resolution / ``SchemaBypassError`` from walking the source parts.
    """
    from spyglass.spikesorting.v2._artifact_intervals import (
        read_artifact_removed_intervals,
    )

    with pytest.raises(
        ValueError, match="must include 'artifact_detection_id'"
    ):
        read_artifact_removed_intervals({})
